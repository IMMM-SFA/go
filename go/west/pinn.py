import logging
import os
from typing import Union, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cloudpickle
from dataclasses import dataclass

from go import configuration
from go.west.linear import model_west_linear_multi
from go.utilities import write_solver_parameters, write_restart_file, get_restart_file

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class PINNConfig:
    """Configuration for the PINN model"""
    hidden_layers: List[int] = None
    learning_rate: float = 1e-3
    max_epochs: int = 10000
    patience: int = 1000
    physics_weight: float = 1.0
    objective_weight: float = 1.0
    constraint_weight: float = 10.0
    batch_size: int = 32
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 256, 256, 256]


class PowerSystemPINN(nn.Module):
    """
    Physics Informed Neural Network for power system optimization.
    
    This neural network learns to predict optimal generation, flows, and other variables
    while satisfying the physical constraints of the power system.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: PINNConfig):
        super(PowerSystemPINN, self).__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build the neural network architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network"""
        return self.network(x)
    
    def predict_variables(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict all optimization variables from inputs.
        
        Args:
            inputs: Input tensor containing demand, renewable availability, etc.
            
        Returns:
            Dictionary containing predicted variables
        """
        outputs = self.forward(inputs)
        
        # Handle 1D output (single sample)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)  # Add batch dimension
        
        # Split outputs into different variable types
        batch_size = outputs.shape[0]
        
        # Calculate variable offsets
        offset = 0
        mwh_size = self.n_generators * self.n_hours
        flow_size = self.n_lines * self.n_hours
        theta_size = self.n_buses * self.n_hours
        slack_size = self.n_buses * self.n_hours
        charge_size = self.n_storage * self.n_hours
        discharge_size = self.n_storage * self.n_hours
        soc_size = self.n_storage * (self.n_hours + 1)
        
        # Calculate total expected size
        total_expected_size = mwh_size + flow_size + theta_size + slack_size + charge_size + discharge_size + soc_size
        
        # Debug: Check if output size matches expected size
        actual_output_size = outputs.shape[1]
        if actual_output_size != total_expected_size:
            print(f"WARNING: Output size mismatch! Expected {total_expected_size}, got {actual_output_size}")
            print(f"System dimensions: n_generators={self.n_generators}, n_lines={self.n_lines}, n_buses={self.n_buses}, n_storage={self.n_storage}, n_hours={self.n_hours}")
            print(f"Individual sizes: mwh={mwh_size}, flow={flow_size}, theta={theta_size}, slack={slack_size}, charge={charge_size}, discharge={discharge_size}, soc={soc_size}")
            
            # Adjust the output to match expected size by padding or truncating
            if actual_output_size < total_expected_size:
                # Pad with zeros
                padding = torch.zeros(batch_size, total_expected_size - actual_output_size, device=outputs.device)
                outputs = torch.cat([outputs, padding], dim=1)
            else:
                # Truncate
                outputs = outputs[:, :total_expected_size]
        
        predictions = {
            'mwh': outputs[:, offset:offset + mwh_size],
            'flow': outputs[:, offset + mwh_size:offset + mwh_size + flow_size],
            'theta': outputs[:, offset + mwh_size + flow_size:offset + mwh_size + flow_size + theta_size],
            'slack': outputs[:, offset + mwh_size + flow_size + theta_size:offset + mwh_size + flow_size + theta_size + slack_size],
            'charge': outputs[:, offset + mwh_size + flow_size + theta_size + slack_size:offset + mwh_size + flow_size + theta_size + slack_size + charge_size],
            'discharge': outputs[:, offset + mwh_size + flow_size + theta_size + slack_size + charge_size:offset + mwh_size + flow_size + theta_size + slack_size + charge_size + discharge_size],
            'soc': outputs[:, offset + mwh_size + flow_size + theta_size + slack_size + charge_size + discharge_size:]
        }

        # Enforce non-negativity on variables that must be >= 0 using a smooth Softplus
        predictions['mwh'] = F.softplus(predictions['mwh'])
        predictions['charge'] = F.softplus(predictions['charge'])
        predictions['discharge'] = F.softplus(predictions['discharge'])
        predictions['soc'] = F.softplus(predictions['soc'])
        
        return predictions
    
    def set_system_dimensions(self, n_generators: int, n_lines: int, n_buses: int, 
                             n_storage: int, n_hours: int):
        """Set the dimensions of the power system"""
        self.n_generators = n_generators
        self.n_lines = n_lines
        self.n_buses = n_buses
        self.n_storage = n_storage
        self.n_hours = n_hours


class PhysicsLoss:
    """Physics-informed loss functions for power system constraints with vectorized operations"""
    
    def __init__(self, system_data: Dict):
        self.system_data = system_data
        # Pre-compute mapping tensors for vectorized operations
        self._prepare_mappings()
    
    def _prepare_mappings(self):
        """Prepare mapping tensors for vectorized operations"""
        # Convert mapping arrays to tensors, handling ScalarParam objects
        def safe_convert_to_tensor(data, dtype=torch.float32):
            if hasattr(data, 'value'):  # Handle ScalarParam objects
                return torch.tensor(data.value, dtype=dtype)
            elif isinstance(data, torch.Tensor):  # Already a tensor
                if data.dtype != dtype:
                    return data.to(dtype)
                return data
            elif isinstance(data, (list, np.ndarray)):
                return torch.tensor(data, dtype=dtype)
            else:
                return torch.tensor(data, dtype=dtype)
        
        # Helper function to safely convert to integer
        def safe_int(data):
            if hasattr(data, 'value'):  # Handle ScalarParam objects
                return int(data.value)
            else:
                return int(data)
        
        self.line_to_bus_map = safe_convert_to_tensor(self.system_data['line_to_bus_map'])
        self.bus_to_unit_map = safe_convert_to_tensor(self.system_data['bus_to_unit_map'])
        self.bus_to_storage_map = safe_convert_to_tensor(self.system_data['bus_to_storage_map'])
        
        # Pre-compute line information for transmission flow
        self.line_from_bus = safe_convert_to_tensor(self.system_data['line_from_bus'], dtype=torch.long)
        self.line_to_bus = safe_convert_to_tensor(self.system_data['line_to_bus'], dtype=torch.long)
        self.reactance = safe_convert_to_tensor(self.system_data['reactance'])
        
        # Store the safe conversion function for use in other methods
        self._safe_int = safe_int
    
    def nodal_balance_loss(self, predictions: Dict[str, torch.Tensor], 
                          demand: torch.Tensor, must_run: torch.Tensor) -> torch.Tensor:
        """
        Vectorized nodal balance loss calculation.
        
        For each bus: generation + slack + must_run - flow = demand + storage_charge - storage_discharge
        """
        try:
            # Convert system data to integers to avoid ScalarParam issues
            n_generators = self._safe_int(self.system_data['n_generators'])
            n_lines = self._safe_int(self.system_data['n_lines'])
            n_buses = self._safe_int(self.system_data['n_buses'])
            n_storage = self._safe_int(self.system_data['n_storage'])
            n_hours = self._safe_int(self.system_data['n_hours'])
            
            mwh = predictions['mwh'].view(-1, n_generators, n_hours)
            flow = predictions['flow'].view(-1, n_lines, n_hours)
            slack = predictions['slack'].view(-1, n_buses, n_hours)
            charge = predictions['charge'].view(-1, n_storage, n_hours)
            discharge = predictions['discharge'].view(-1, n_storage, n_hours)
            
            # Get device from predictions and move mapping tensors if needed
            device = predictions['mwh'].device
            if self.line_to_bus_map.device != device:
                self.line_to_bus_map = self.line_to_bus_map.to(device)
                self.bus_to_unit_map = self.bus_to_unit_map.to(device)
                self.bus_to_storage_map = self.bus_to_storage_map.to(device)
            
            # Vectorized power flow calculation using einsum
            # flow: [batch, n_lines, n_hours], line_to_bus_map: [n_lines, n_buses]
            # Result: [batch, n_buses, n_hours]
            power_flow = torch.einsum('blh,lz->bzh', flow, self.line_to_bus_map)
            
            # Vectorized generation calculation
            # mwh: [batch, n_generators, n_hours], bus_to_unit_map: [n_generators, n_buses]
            # Result: [batch, n_buses, n_hours]
            generation = torch.einsum('bgh,gz->bzh', mwh, self.bus_to_unit_map)
            
            # Vectorized storage calculation
            # charge/discharge: [batch, n_storage, n_hours], bus_to_storage_map: [n_storage, n_buses]
            # Result: [batch, n_buses, n_hours]
            storage_charge = torch.einsum('bsh,sz->bzh', charge, self.bus_to_storage_map)
            storage_discharge = torch.einsum('bsh,sz->bzh', discharge, self.bus_to_storage_map)
            
            # Nodal balance constraint
            balance = generation + slack + must_run - power_flow - demand - storage_charge + storage_discharge
            
            return torch.mean(torch.square(balance))
        except Exception as e:
            print(f"Error in vectorized nodal_balance_loss: {e}")
            print(f"predictions keys: {list(predictions.keys())}")
            for key, value in predictions.items():
                print(f"  {key}: {value.shape}")
            raise
    
    def transmission_flow_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Vectorized transmission line flow loss calculation"""
        try:
            # Convert system data to integers
            n_lines = self._safe_int(self.system_data['n_lines'])
            n_buses = self._safe_int(self.system_data['n_buses'])
            n_hours = self._safe_int(self.system_data['n_hours'])
            
            flow = predictions['flow'].view(-1, n_lines, n_hours)
            theta = predictions['theta'].view(-1, n_buses, n_hours)
            
            # Get device and move tensors if needed
            device = predictions['flow'].device
            if self.line_from_bus.device != device:
                self.line_from_bus = self.line_from_bus.to(device)
                self.line_to_bus = self.line_to_bus.to(device)
                self.reactance = self.reactance.to(device)
            
            # Vectorized flow calculation
            # theta: [batch, n_buses, n_hours]
            # line_from_bus, line_to_bus: [n_lines]
            # reactance: [n_lines]
            
            # Get theta values for from and to buses
            theta_from = theta[:, self.line_from_bus, :]  # [batch, n_lines, n_hours]
            theta_to = theta[:, self.line_to_bus, :]      # [batch, n_lines, n_hours]
            
            # Calculate flow: (theta_from - theta_to) / reactance
            flow_calculated = (theta_from - theta_to) / self.reactance.unsqueeze(0).unsqueeze(-1)
            
            # Compare with predicted flow
            return torch.mean(torch.square(flow - flow_calculated))
            
        except Exception as e:
            print(f"Error in vectorized transmission_flow_loss: {e}")
            print(f"line_from_bus: {self.system_data['line_from_bus']}")
            print(f"line_to_bus: {self.system_data['line_to_bus']}")
            raise
    
    def capacity_constraints_loss(self, predictions: Dict[str, torch.Tensor], 
                                max_capacity: torch.Tensor) -> torch.Tensor:
        """Calculate loss for generator capacity constraints"""
        # Convert system data to integers
        n_generators = self._safe_int(self.system_data['n_generators'])
        n_hours = self._safe_int(self.system_data['n_hours'])
        
        mwh = predictions['mwh'].view(-1, n_generators, n_hours)
        
        # Maximum capacity constraint: mwh <= max_capacity
        capacity_violation = torch.relu(mwh - max_capacity)
        
        # Minimum capacity constraint: mwh >= 0
        min_capacity_violation = torch.relu(-mwh)
        
        return torch.mean(torch.square(capacity_violation)) + torch.mean(torch.square(min_capacity_violation))
    
    def storage_constraints_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for storage constraints"""
        try:
            # Convert system data to integers
            n_storage = self._safe_int(self.system_data['n_storage'])
            n_hours = self._safe_int(self.system_data['n_hours'])
            
            # Get device from predictions
            device = predictions['charge'].device
            
            # Ensure storage parameters are tensors on the correct device
            charge_rate = self.system_data['charge_rate']
            discharge_rate = self.system_data['discharge_rate']
            max_soc = self.system_data['max_soc']
            min_soc = self.system_data['min_soc']
            charge_eff = self.system_data['charge_eff']
            discharge_eff = self.system_data['discharge_eff']
            
            # Convert to tensors if they aren't already and move to correct device
            if not isinstance(charge_rate, torch.Tensor):
                charge_rate = torch.FloatTensor([float(charge_rate)] if n_storage == 1 else [float(x) for x in charge_rate])
            charge_rate = charge_rate.to(device)
            
            if not isinstance(discharge_rate, torch.Tensor):
                discharge_rate = torch.FloatTensor([float(discharge_rate)] if n_storage == 1 else [float(x) for x in discharge_rate])
            discharge_rate = discharge_rate.to(device)
            
            if not isinstance(max_soc, torch.Tensor):
                max_soc = torch.FloatTensor([float(max_soc)] if n_storage == 1 else [float(x) for x in max_soc])
            max_soc = max_soc.to(device)
            
            if not isinstance(min_soc, torch.Tensor):
                min_soc = torch.FloatTensor([float(min_soc)] if n_storage == 1 else [float(x) for x in min_soc])
            min_soc = min_soc.to(device)
            
            if not isinstance(charge_eff, torch.Tensor):
                charge_eff = torch.FloatTensor([float(charge_eff)] if n_storage == 1 else [float(x) for x in charge_eff])
            charge_eff = charge_eff.to(device)
            
            if not isinstance(discharge_eff, torch.Tensor):
                discharge_eff = torch.FloatTensor([float(discharge_eff)] if n_storage == 1 else [float(x) for x in discharge_eff])
            discharge_eff = discharge_eff.to(device)
            
            charge = predictions['charge'].view(-1, n_storage, n_hours)
            discharge = predictions['discharge'].view(-1, n_storage, n_hours)
            soc = predictions['soc'].view(-1, n_storage, n_hours + 1)
            
            # Charge rate constraints
            charge_rate_violation = torch.relu(charge - charge_rate.unsqueeze(-1))
            
            # Discharge rate constraints
            discharge_rate_violation = torch.relu(discharge - discharge_rate.unsqueeze(-1))
            
            # SOC constraints
            soc_max_violation = torch.relu(soc - max_soc.unsqueeze(-1))
            soc_min_violation = torch.relu(min_soc.unsqueeze(-1) - soc)
            
            # SOC balance constraint
            soc_balance = soc[:, :, 1:] - (soc[:, :, :-1] + 
                                          charge * charge_eff.unsqueeze(-1) - 
                                          discharge / discharge_eff.unsqueeze(-1))
            
            # Simultaneous charge/discharge prevention
            sim_charge_discharge_violation = torch.relu(
                discharge - (discharge_rate.unsqueeze(-1) - 
                           (discharge_rate.unsqueeze(-1) / charge_rate.unsqueeze(-1)) * charge)
            )
            
            return (torch.mean(torch.square(charge_rate_violation)) + 
                    torch.mean(torch.square(discharge_rate_violation)) + 
                    torch.mean(torch.square(soc_max_violation)) + 
                    torch.mean(torch.square(soc_min_violation)) + 
                    torch.mean(torch.square(soc_balance)) +
                    torch.mean(torch.square(sim_charge_discharge_violation)))
        except Exception as e:
            print(f"Error in storage_constraints_loss: {e}")
            print(f"system_data keys: {list(self.system_data.keys())}")
            for key in ['charge_rate', 'discharge_rate', 'max_soc', 'min_soc', 'charge_eff', 'discharge_eff']:
                if key in self.system_data:
                    print(f"  {key}: {type(self.system_data[key])} - {self.system_data[key]}")
            raise
    
    def ramp_rate_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for generator ramp rate constraints"""
        try:
            # Convert system data to integers
            n_generators = self._safe_int(self.system_data['n_generators'])
            n_hours = self._safe_int(self.system_data['n_hours'])
            
            # Get device from predictions
            device = predictions['mwh'].device
            
            # Get ramp rates and move to device
            ramp_rates = self.system_data['ramp_rates']
            if not isinstance(ramp_rates, torch.Tensor):
                ramp_rates = torch.FloatTensor([float(x) for x in ramp_rates])
            ramp_rates = ramp_rates.to(device)
            
            mwh = predictions['mwh'].view(-1, n_generators, n_hours)
            
            # Ramp up constraint: mwh[j,i] - mwh[j,i-1] <= ramp[j]
            ramp_up_violation = torch.relu(mwh[:, :, 1:] - mwh[:, :, :-1] - ramp_rates.unsqueeze(-1))
            
            # Ramp down constraint: mwh[j,i-1] - mwh[j,i] <= ramp[j]
            ramp_down_violation = torch.relu(mwh[:, :, :-1] - mwh[:, :, 1:] - ramp_rates.unsqueeze(-1))
            
            return torch.mean(torch.square(ramp_up_violation)) + torch.mean(torch.square(ramp_down_violation))
            
        except Exception as e:
            print(f"Error in ramp_rate_loss: {e}")
            return torch.tensor(0.0, device=predictions['mwh'].device)
    
    def line_flow_limits_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for transmission line flow limits"""
        try:
            # Convert system data to integers
            n_lines = self._safe_int(self.system_data['n_lines'])
            n_hours = self._safe_int(self.system_data['n_hours'])
            
            # Get device from predictions
            device = predictions['flow'].device
            
            # Get flow limits and move to device
            flow_limits = self.system_data['flow_limits']
            if not isinstance(flow_limits, torch.Tensor):
                flow_limits = torch.FloatTensor([float(x) for x in flow_limits])
            flow_limits = flow_limits.to(device)
            
            flow = predictions['flow'].view(-1, n_lines, n_hours)
            
            # Flow limits: -FlowLim[l] <= Flow[l,i] <= FlowLim[l]
            flow_upper_violation = torch.relu(flow - flow_limits.unsqueeze(-1))
            flow_lower_violation = torch.relu(-flow - flow_limits.unsqueeze(-1))
            
            return torch.mean(torch.square(flow_upper_violation)) + torch.mean(torch.square(flow_lower_violation))
            
        except Exception as e:
            print(f"Error in line_flow_limits_loss: {e}")
            return torch.tensor(0.0, device=predictions['flow'].device)
    
    def hydro_production_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for hydro production constraints"""
        try:
            # Convert system data to integers
            n_generators = self._safe_int(self.system_data['n_generators'])
            n_hours = self._safe_int(self.system_data['n_hours'])
            
            # Get device from predictions
            device = predictions['mwh'].device
            
            # Get hydro total limits and hydro generator indices
            hydro_total_limits = self.system_data['hydro_total_limits']
            hydro_generator_indices = self.system_data['hydro_generator_indices']
            
            if not isinstance(hydro_total_limits, torch.Tensor):
                hydro_total_limits = torch.FloatTensor([float(x) for x in hydro_total_limits])
            hydro_total_limits = hydro_total_limits.to(device)
            
            mwh = predictions['mwh'].view(-1, n_generators, n_hours)
            
            # Daily hydro production constraint: sum(mwh[j,i] for i in hours) <= HorizonHydro_TOTAL[j]
            hydro_generation = mwh[:, hydro_generator_indices, :]  # [batch, n_hydro, n_hours]
            daily_hydro_production = torch.sum(hydro_generation, dim=2)  # [batch, n_hydro]
            
            hydro_total_violation = torch.relu(daily_hydro_production - hydro_total_limits.unsqueeze(0))
            
            return torch.mean(torch.square(hydro_total_violation))
            
        except Exception as e:
            print(f"Error in hydro_production_loss: {e}")
            return torch.tensor(0.0, device=predictions['mwh'].device)
    
    def renewable_capacity_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for renewable capacity constraints"""
        try:
            # Convert system data to integers
            n_generators = self._safe_int(self.system_data['n_generators'])
            n_hours = self._safe_int(self.system_data['n_hours'])
            
            # Get device from predictions
            device = predictions['mwh'].device
            
            # Get renewable capacity limits and generator indices
            renewable_capacity_limits = self.system_data['renewable_capacity_limits']  # [n_renewable, n_hours]
            renewable_generator_indices = self.system_data['renewable_generator_indices']
            
            if not isinstance(renewable_capacity_limits, torch.Tensor):
                renewable_capacity_limits = torch.FloatTensor(renewable_capacity_limits)
            renewable_capacity_limits = renewable_capacity_limits.to(device)
            
            mwh = predictions['mwh'].view(-1, n_generators, n_hours)
            
            # Renewable capacity constraints: mwh[j,i] <= renewable_capacity[j,i]
            renewable_generation = mwh[:, renewable_generator_indices, :]  # [batch, n_renewable, n_hours]
            renewable_capacity_violation = torch.relu(renewable_generation - renewable_capacity_limits.unsqueeze(0))
            
            return torch.mean(torch.square(renewable_capacity_violation))
            
        except Exception as e:
            print(f"Error in renewable_capacity_loss: {e}")
            return torch.tensor(0.0, device=predictions['mwh'].device)
    
    def bus_angle_reference_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for bus angle reference constraint"""
        try:
            # Convert system data to integers
            n_buses = self._safe_int(self.system_data['n_buses'])
            n_hours = self._safe_int(self.system_data['n_hours'])
            
            # Get device from predictions
            device = predictions['theta'].device
            
            # Get reference bus index
            reference_bus_index = self.system_data['reference_bus_index']
            
            theta = predictions['theta'].view(-1, n_buses, n_hours)
            
            # Reference bus angle constraint: Theta[reference_bus, i] == 0
            reference_bus_angle = theta[:, reference_bus_index, :]  # [batch, n_hours]
            reference_violation = torch.square(reference_bus_angle)
            
            return torch.mean(reference_violation)
            
        except Exception as e:
            print(f"Error in bus_angle_reference_loss: {e}")
            return torch.tensor(0.0, device=predictions['theta'].device)
    
    def objective_loss(self, predictions: Dict[str, torch.Tensor], 
                      fuel_prices: torch.Tensor, heat_rates: torch.Tensor) -> torch.Tensor:
        """Calculate the objective function (total system cost)"""
        # Convert system data to integers
        n_generators = self._safe_int(self.system_data['n_generators'])
        n_buses = self._safe_int(self.system_data['n_buses'])
        n_lines = self._safe_int(self.system_data['n_lines'])
        n_storage = self._safe_int(self.system_data['n_storage'])
        n_hours = self._safe_int(self.system_data['n_hours'])
        
        mwh = predictions['mwh'].view(-1, n_generators, n_hours)
        slack = predictions['slack'].view(-1, n_buses, n_hours)
        flow = predictions['flow'].view(-1, n_lines, n_hours)
        charge = predictions['charge'].view(-1, n_storage, n_hours)
        discharge = predictions['discharge'].view(-1, n_storage, n_hours)
        
        # Generation cost
        gen_cost = torch.sum(mwh * heat_rates.unsqueeze(-1) * fuel_prices.unsqueeze(-1), dim=(1, 2))
        
        # Slack cost (penalty for unmet demand and excess generation)
        positive_slack_cost = torch.sum(torch.relu(slack) * 2000, dim=(1, 2))  # Unmet demand penalty
        negative_slack_cost = torch.sum(torch.relu(-slack) * 1000, dim=(1, 2))  # Excess generation penalty
        slack_cost = positive_slack_cost + negative_slack_cost
        
        # Storage costs
        storage_cost = torch.sum(charge * 0.001 + discharge * 0.001, dim=(1, 2))
        
        # Flow penalty
        flow_penalty = torch.sum(torch.abs(flow) * 0.01, dim=(1, 2))
        
        # Slack regularization to prevent extreme values
        slack_regularization = torch.sum(torch.square(slack) * 0.1, dim=(1, 2))
        
        total_cost = gen_cost + slack_cost + storage_cost + flow_penalty + slack_regularization
        
        return torch.mean(total_cost)


class PINNTrainer:
    """Trainer for the Physics Informed Neural Network"""
    
    def __init__(self, model: PowerSystemPINN, physics_loss: PhysicsLoss, config: PINNConfig):
        self.model = model
        self.physics_loss = physics_loss
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=config.patience//2)
        
    def train_step(self, inputs: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        try:
            
            self.optimizer.zero_grad()
            
            # Forward pass
            # print("train_step: calling predict_variables")
            predictions = self.model.predict_variables(inputs)
            # print(f"train_step: predictions keys = {list(predictions.keys())}")
            
            # Calculate losses
            # print("train_step: calling nodal_balance_loss")
            physics_loss = self.physics_loss.nodal_balance_loss(predictions, targets['demand'], targets['must_run'])
            # print("train_step: calling transmission_flow_loss")
            physics_loss += self.physics_loss.transmission_flow_loss(predictions)
            # print("train_step: calling capacity_constraints_loss")
            physics_loss += self.physics_loss.capacity_constraints_loss(predictions, targets['max_capacity'])
            # print("train_step: calling storage_constraints_loss")
            physics_loss += self.physics_loss.storage_constraints_loss(predictions)
            # print("train_step: calling ramp_rate_loss")
            physics_loss += self.physics_loss.ramp_rate_loss(predictions)
            # print("train_step: calling line_flow_limits_loss")
            physics_loss += self.physics_loss.line_flow_limits_loss(predictions)
            # print("train_step: calling hydro_production_loss")
            physics_loss += self.physics_loss.hydro_production_loss(predictions)
            # print("train_step: calling renewable_capacity_loss")
            physics_loss += self.physics_loss.renewable_capacity_loss(predictions)
            # print("train_step: calling bus_angle_reference_loss")
            physics_loss += self.physics_loss.bus_angle_reference_loss(predictions)
            
            # print("train_step: calling objective_loss")
            objective_loss = self.physics_loss.objective_loss(predictions, targets['fuel_prices'], targets['heat_rates'])
            
            # Total loss
            total_loss = (self.config.physics_weight * physics_loss + 
                         self.config.objective_weight * objective_loss)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            return {
                'total_loss': total_loss.item(),
                'physics_loss': physics_loss.item(),
                'objective_loss': objective_loss.item()
            }
        except Exception as e:
            print(f"Error in train_step: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> List[Dict[str, float]]:
        """Train the model"""
        history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            epoch_losses = []
            
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.model.device)
                batch_targets = {k: v.to(self.model.device) for k, v in batch_targets.items()}
                
                losses = self.train_step(batch_inputs, batch_targets)
                epoch_losses.append(losses)
            
            # Average losses for the epoch
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            history.append(avg_losses)
            
            # Learning rate scheduling
            self.scheduler.step(avg_losses['total_loss'])
            
            # Early stopping
            if avg_losses['total_loss'] < best_loss:
                best_loss = avg_losses['total_loss']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Total Loss = {avg_losses['total_loss']:.6f}, "
                      f"Physics Loss = {avg_losses['physics_loss']:.6f}, "
                      f"Objective Loss = {avg_losses['objective_loss']:.6f}")
        
        return history


def west_linear_multi_pinn(
    config_file: str,
    pinn_config: PINNConfig = None,
    n_days: int = 365,
    restart_day: Union[None, int] = None,
    save_restart_file: bool = True,
    break_run: bool = False,
    restart_write_frequency: int = 10,
    **kwargs
):
    """
    Physics Informed Neural Network version of the West Linear Multi Model.
    
    This function replaces the traditional optimization solver with a neural network
    that learns to satisfy the physical constraints of the power system.
    
    :param config_file:         The configuration file to use.
    :type config_file:          str
    
    :param pinn_config:         Configuration for the PINN model.
    :type pinn_config:          PINNConfig
    
    :param n_days:              The number of days to process through.
    :type n_days:               int
    
    :param restart_day:         Day of the restart file to use.
    :type restart_day:          Union[None, int]
    
    :param save_restart_file:   If True, save a restart file after each timestep.
    :type save_restart_file:    bool
    
    :param break_run:           If True, run will break after one day iteration.
    :type break_run:            bool
    
    :param restart_write_frequency: Write a restart file after this many days.
    :type restart_write_frequency:  int
    
    :param kwargs:              Additional keyword arguments.
    :type kwargs:               dict
    """
    
    logger = logging.getLogger(__name__)
    
    if pinn_config is None:
        pinn_config = PINNConfig()
    
    logger.info("Prepare PINN simulation")
    
    # Read configuration
    config = configuration.generate_config(config_file=config_file, **kwargs)
    
    # Read input files
    df_generators = pd.read_csv(config.generator_parameters_file, header=0)
    df_thermal = pd.read_csv(config.thermal_generators_file, header=0)
    df_loss_dict = np.load(config.generator_outage_file, allow_pickle=True).item()
    df_losses = pd.read_csv(config.lost_capacity_file, header=0, index_col=0)
    
    # Extract nuclear
    nucs = df_thermal[df_thermal['Fuel'] == 'NUC (Nuclear)'].copy()
    
    # Get restart file
    restart_file = get_restart_file(
        dir=config.restart_file_directory,
        day=restart_day,
    )
    
    # Initialize or load from restart
    if restart_file is None:
        start_day = 1
        
        # Create Pyomo model to get system dimensions and data
        go_model = model_west_linear_multi()
        instance = go_model.create_instance(config.dat_file)
        
        # Extract system dimensions
        n_generators = len(instance.Generators)
        n_lines = len(instance.lines)
        n_buses = len(instance.buses)
        n_storage = len(instance.Storage)
        n_hours = int(instance.HorizonHours.value) if hasattr(instance.HorizonHours, 'value') else int(instance.HorizonHours)
        
        # Prepare system data for PINN
        system_data = prepare_system_data(instance, df_generators, df_thermal)
        
        # Initialize PINN model
        input_dim = n_buses * n_hours  # Demand at each bus for each hour
        output_dim = (n_generators * n_hours +  # Generation
                     n_lines * n_hours +        # Flows
                     n_buses * n_hours +        # Voltage angles
                     n_buses * n_hours +        # Slack
                     n_storage * n_hours +      # Charge
                     n_storage * n_hours +      # Discharge
                     n_storage * (n_hours + 1)) # SOC (including initial state)
        
        pinn_model = PowerSystemPINN(input_dim, output_dim, pinn_config)
        pinn_model.set_system_dimensions(n_generators, n_lines, n_buses, n_storage, n_hours)
        
        # Initialize physics loss
        physics_loss = PhysicsLoss(system_data)
        
        # Initialize trainer
        trainer = PINNTrainer(pinn_model, physics_loss, pinn_config)
        
        # Initialize output storage
        mwh = []
        flow = []
        slack = []
        vlt_angle = []
        charge = []
        discharge = []
        SoC = []
        
    else:
        logger.info(f"Loading from restart file: {restart_file}")
        
        with open(restart_file, "rb") as f:
            restart_data = cloudpickle.load(f)
        
        pinn_model = restart_data["pinn_model"]
        system_data = restart_data["system_data"]
        physics_loss = restart_data["physics_loss"]
        trainer = restart_data["trainer"]
        instance = restart_data["instance"]
        
        mwh = restart_data["mwh"]
        flow = restart_data["flow"]
        slack = restart_data["slack"]
        vlt_angle = restart_data["vlt_angle"]
        charge = restart_data["charge"]
        discharge = restart_data["discharge"]
        SoC = restart_data["SoC"]
        
        start_day = restart_data["day"] + 1
    
    # Main simulation loop
    for day in range(start_day, n_days + 1):
        logger.info(f"Day {day}: Processing with PINN")
        
        # Update instance data for the current day
        update_instance_for_day(instance, day, df_losses, df_loss_dict, nucs)
        
        # Prepare input data for the current day
        inputs, targets = prepare_day_data(instance, day, system_data)
        
        # Convert to tensors
        inputs_tensor = torch.FloatTensor(inputs).to(pinn_model.device)
        targets_tensor = {k: torch.FloatTensor(v).to(pinn_model.device) for k, v in targets.items()}
        
        # For single sample training, we'll train directly without DataLoader
        logger.info(f"Day {day}: Training PINN")
        
        # Train the PINN for this day (single sample)
        history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(pinn_config.max_epochs):
            # Single training step
            losses = trainer.train_step(inputs_tensor, targets_tensor)
            history.append(losses)
            
            # Learning rate scheduling
            trainer.scheduler.step(losses['total_loss'])
            
            # Early stopping
            if losses['total_loss'] < best_loss:
                best_loss = losses['total_loss']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= pinn_config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Total Loss = {losses['total_loss']:.6f}, "
                      f"Physics Loss = {losses['physics_loss']:.6f}, "
                      f"Objective Loss = {losses['objective_loss']:.6f}")
        
        # Get predictions
        with torch.no_grad():
            predictions = pinn_model.predict_variables(inputs_tensor)
        
        # Store results
        store_day_results(predictions, day, mwh, flow, slack, vlt_angle, charge, discharge, SoC, instance, df_generators)
        
        # Save restart file
        if save_restart_file and day % restart_write_frequency == 0:
            restart_data = {
                "pinn_model": pinn_model,
                "system_data": system_data,
                "physics_loss": physics_loss,
                "trainer": trainer,
                "instance": instance,
                "mwh": mwh,
                "flow": flow,
                "slack": slack,
                "vlt_angle": vlt_angle,
                "charge": charge,
                "discharge": discharge,
                "SoC": SoC,
                "day": day
            }
            
            new_restart_file_path = write_restart_file(
                dir=config.restart_file_directory,
                day=day,
                restart_data=restart_data,
            )
            
            if new_restart_file_path is not None:
                logger.info(f'Day {day}: Restart file written to {new_restart_file_path}.')
        
        logger.info(f'Day {day} completed.')
        
        if break_run:
            break
    
    # Save final results and trained model
    model_dir = save_final_results(config, mwh, flow, slack, vlt_angle, charge, discharge, SoC, pinn_model, system_data)
    
    return day


def prepare_system_data(instance, df_generators, df_thermal):
    """Prepare system data for the PINN"""
    
    def safe_get_value(param, *indices):
        """Safely get parameter value, handling both Pyomo parameters and direct values"""
        try:
            if hasattr(param, 'value'):
                return param.value
            else:
                return param
        except:
            return 0.0
    
    # Helper function to safely convert to integer
    def safe_int(data):
        if hasattr(data, 'value'):  # Handle ScalarParam objects
            return int(data.value)
        else:
            return int(data)
    
    system_data = {
        'n_generators': len(instance.Generators),
        'n_lines': len(instance.lines),
        'n_buses': len(instance.buses),
        'n_storage': len(instance.Storage),
        'n_hours': safe_int(instance.HorizonHours),
        
        # Mappings
        'line_to_bus_map': torch.FloatTensor([[safe_get_value(instance.LinetoBusMap[l, z]) for z in instance.buses] for l in instance.lines]),
        'bus_to_unit_map': torch.FloatTensor([[safe_get_value(instance.BustoUnitMap[j, z]) for z in instance.buses] for j in instance.Generators]),
        'bus_to_storage_map': torch.FloatTensor([[safe_get_value(instance.BustoStorageMap[j, z]) for z in instance.buses] for j in instance.Storage]),
        
        # Line data
        'reactance': torch.FloatTensor([safe_get_value(instance.Reactance[l]) for l in instance.lines]),
        'flow_limits': torch.FloatTensor([safe_get_value(instance.FlowLim[l]) for l in instance.lines]),
        
        # Extract line-to-bus mappings from the LinetoBusMap
        'line_from_bus': [],
        'line_to_bus': [],
        
        # Storage data
        'charge_rate': torch.FloatTensor([safe_get_value(instance.charge_rate[j]) for j in instance.Storage]),
        'discharge_rate': torch.FloatTensor([safe_get_value(instance.discharge_rate[j]) for j in instance.Storage]),
        'max_soc': torch.FloatTensor([safe_get_value(instance.max_SoC[j]) for j in instance.Storage]),
        'min_soc': torch.FloatTensor([safe_get_value(instance.min_SoC[j]) for j in instance.Storage]),
        'charge_eff': torch.FloatTensor([safe_get_value(instance.charge_eff[j]) for j in instance.Storage]),
        'discharge_eff': torch.FloatTensor([safe_get_value(instance.discharge_eff[j]) for j in instance.Storage]),
        
        # Generator ramp rates
        'ramp_rates': torch.FloatTensor([safe_get_value(instance.ramp[j]) for j in instance.Generators]),
        
        # Hydro data
        'hydro_total_limits': torch.FloatTensor([safe_get_value(instance.HorizonHydro_TOTAL[j]) for j in instance.Hydro]),
        'hydro_generator_indices': [list(instance.Generators).index(j) for j in instance.Hydro],
        
        # Renewable data
        'renewable_capacity_limits': torch.FloatTensor([
            [safe_get_value(instance.HorizonSolar[j, i]) for i in range(1, safe_int(instance.HorizonHours) + 1)] 
            for j in instance.Solar
        ] + [
            [safe_get_value(instance.HorizonWind[j, i]) for i in range(1, safe_int(instance.HorizonHours) + 1)] 
            for j in instance.Wind
        ] + [
            [safe_get_value(instance.HorizonOffshoreWind[j, i]) for i in range(1, safe_int(instance.HorizonHours) + 1)] 
            for j in instance.OffshoreWind
        ]),
        'renewable_generator_indices': [list(instance.Generators).index(j) for j in list(instance.Solar) + list(instance.Wind) + list(instance.OffshoreWind)],
        
        # Reference bus (assuming bus_100011 as reference)
        'reference_bus_index': list(instance.buses).index('bus_100011') if 'bus_100011' in instance.buses else 0,
    }
    
    # Extract line-to-bus mappings from LinetoBusMap
    for l in instance.lines:
        from_bus = None
        to_bus = None
        for z in instance.buses:
            if safe_get_value(instance.LinetoBusMap[l, z]) == 1:  # From bus
                from_bus = z
            elif safe_get_value(instance.LinetoBusMap[l, z]) == -1:  # To bus
                to_bus = z
        
        # Convert bus names to indices
        bus_list = list(instance.buses)
        if from_bus is not None and to_bus is not None:
            system_data['line_from_bus'].append(bus_list.index(from_bus))
            system_data['line_to_bus'].append(bus_list.index(to_bus))
        else:
            # Fallback: use first two buses
            system_data['line_from_bus'].append(0)
            system_data['line_to_bus'].append(1 if len(bus_list) > 1 else 0)
    
    return system_data


def update_instance_for_day(instance, day, df_losses, df_loss_dict, nucs):
    """Update the Pyomo instance with data for the current day"""
    n_hours = int(instance.HorizonHours.value) if hasattr(instance.HorizonHours, 'value') else int(instance.HorizonHours)
    horizon_hours_series = range(1, n_hours + 1)
    
    # Update demand
    for z in instance.buses:
        for i in horizon_hours_series:
            instance.HorizonDemand[z, i] = instance.SimDemand[z, (day - 1) * 24 + i]
    
    # Update hydro data
    for z in instance.Hydro:
        instance.HorizonHydro_MAX[z] = instance.SimHydro_MAX[z, day]
        instance.HorizonHydro_MIN[z] = instance.SimHydro_MIN[z, day]
        instance.HorizonHydro_TOTAL[z] = instance.SimHydro_TOTAL[z, day]
    
    # Update renewable data
    for z in instance.Solar:
        for i in horizon_hours_series:
            instance.HorizonSolar[z, i] = instance.SimSolar[z, (day - 1) * 24 + i]
    
    for z in instance.Wind:
        for i in horizon_hours_series:
            instance.HorizonWind[z, i] = instance.SimWind[z, (day - 1) * 24 + i]
    
    for z in instance.OffshoreWind:
        for i in horizon_hours_series:
            instance.HorizonOffshoreWind[z, i] = instance.SimOffshoreWind[z, (day - 1) * 24 + i]
    
    # Update fuel prices
    for z in instance.Thermal:
        instance.FuelPrice[z] = instance.SimFuelPrice[z, day]
    
    # Update outage data
    for z in instance.Outage:
        for i in horizon_hours_series:
            if (z, i) in instance.HorizonGenLimit:
                instance.HorizonGenLimit[z, i] = instance.SimGenLimit[z, (day - 1) * 24 + i]
    
    for z in instance.buses:
        for i in horizon_hours_series:
            if (z, i) in instance.HorizonMustrunLimit:
                instance.HorizonMustrunLimit[z, i] = instance.SimMustrunLimit[z, (day - 1) * 24 + i]
    
    # Apply capacity losses (simplified version)
    # This would need to be expanded to match the original implementation
    for z in instance.Gas:
        for i in horizon_hours_series:
            if (z, i) in instance.HorizonGenLimit:
                current_value = instance.HorizonGenLimit[z, i].value if hasattr(instance.HorizonGenLimit[z, i], 'value') else instance.HorizonGenLimit[z, i]
                instance.HorizonGenLimit[z, i] = max(0, current_value - 0.01)  # Simplified loss
    
    for z in instance.Coal:
        for i in horizon_hours_series:
            if (z, i) in instance.HorizonGenLimit:
                current_value = instance.HorizonGenLimit[z, i].value if hasattr(instance.HorizonGenLimit[z, i], 'value') else instance.HorizonGenLimit[z, i]
                instance.HorizonGenLimit[z, i] = max(0, current_value - 0.01)  # Simplified loss


def prepare_day_data(instance, day, system_data):
    """Prepare input and target data for a specific day"""
    n_hours = system_data['n_hours']
    
    def safe_get_value(param):
        """Safely get parameter value"""
        try:
            if hasattr(param, 'value'):
                return param.value
            else:
                return param
        except:
            return 0.0
    
    # Prepare inputs (demand at each bus for each hour)
    inputs = []
    for z in instance.buses:
        for i in range(1, n_hours + 1):
            inputs.append(safe_get_value(instance.SimDemand[z, (day - 1) * 24 + i]))
    
    # Prepare targets with index validation
    targets = {
        'demand': np.array([[safe_get_value(instance.SimDemand[z, (day - 1) * 24 + i]) 
                           for i in range(1, n_hours + 1)] for z in instance.buses]),
        'must_run': np.array([[safe_get_value(instance.SimMustrunLimit[z, (day - 1) * 24 + i]) 
                             for i in range(1, n_hours + 1)] for z in instance.buses]),
        'max_capacity': np.array([[safe_get_value(instance.HorizonGenLimit[j, i]) if (j, i) in instance.HorizonGenLimit else 0.0
                                 for i in range(1, n_hours + 1)] for j in instance.Generators]),
        'heat_rates': np.array([safe_get_value(instance.heat_rate[j]) for j in instance.Generators]),
    }
    
    # Create fuel prices for all generators (zero for non-thermal)
    fuel_prices = []
    for j in instance.Generators:
        if j in instance.Thermal:
            fuel_prices.append(safe_get_value(instance.SimFuelPrice[j, day]))
        else:
            fuel_prices.append(0.0)  # Zero fuel price for renewables
    targets['fuel_prices'] = np.array(fuel_prices)
    
    return np.array(inputs), targets


def store_day_results(predictions, day, mwh, flow, slack, vlt_angle, charge, discharge, SoC, instance, df_generators):
    """Store the results for a specific day"""
    predictions_np = {k: v.cpu().numpy() for k, v in predictions.items()}
    
    # Reshape predictions
    n_generators = len(instance.Generators)
    n_lines = len(instance.lines)
    n_buses = len(instance.buses)
    n_storage = len(instance.Storage)
    n_hours = int(instance.HorizonHours.value) if hasattr(instance.HorizonHours, 'value') else int(instance.HorizonHours)
    
    mwh_pred = predictions_np['mwh'].reshape(-1, n_generators, n_hours)
    flow_pred = predictions_np['flow'].reshape(-1, n_lines, n_hours)
    theta_pred = predictions_np['theta'].reshape(-1, n_buses, n_hours)
    slack_pred = predictions_np['slack'].reshape(-1, n_buses, n_hours)
    charge_pred = predictions_np['charge'].reshape(-1, n_storage, n_hours)
    discharge_pred = predictions_np['discharge'].reshape(-1, n_storage, n_hours)
    soc_pred = predictions_np['soc'].reshape(-1, n_storage, n_hours + 1)
    
    # Store results in the same format as the original implementation
    for j in range(n_generators):
        gen_name = list(instance.Generators)[j]
        gen_heatrate = df_generators[df_generators['name'] == gen_name]['heat_rate'].values[0]
        
        for i in range(n_hours):
            if gen_name in instance.Gas:
                mwh.append((gen_name, 'Gas', i + 1 + ((day - 1) * 24), mwh_pred[0, j, i]))
            elif gen_name in instance.Coal:
                mwh.append((gen_name, 'Coal', i + 1 + ((day - 1) * 24), mwh_pred[0, j, i]))
            elif gen_name in instance.Oil:
                mwh.append((gen_name, 'Oil', i + 1 + ((day - 1) * 24), mwh_pred[0, j, i]))
            elif gen_name in instance.Hydro:
                mwh.append((gen_name, 'Hydro', i + 1 + ((day - 1) * 24), mwh_pred[0, j, i]))
            elif gen_name in instance.Solar:
                mwh.append((gen_name, 'Solar', i + 1 + ((day - 1) * 24), mwh_pred[0, j, i]))
            elif gen_name in instance.Wind:
                mwh.append((gen_name, 'Wind', i + 1 + ((day - 1) * 24), mwh_pred[0, j, i]))
            elif gen_name in instance.OffshoreWind:
                mwh.append((gen_name, 'OffshoreWind', i + 1 + ((day - 1) * 24), mwh_pred[0, j, i]))
            elif gen_name in instance.Biomass:
                mwh.append((gen_name, 'Biomass', i + 1 + ((day - 1) * 24), mwh_pred[0, j, i]))
            elif gen_name in instance.Geothermal:
                mwh.append((gen_name, 'Geothermal', i + 1 + ((day - 1) * 24), mwh_pred[0, j, i]))
    
    # Store flow results
    for l in range(n_lines):
        line_name = list(instance.lines)[l]
        for i in range(n_hours):
            flow.append((line_name, i + 1 + ((day - 1) * 24), flow_pred[0, l, i]))
    
    # Store slack results
    for z in range(n_buses):
        bus_name = list(instance.buses)[z]
        for i in range(n_hours):
            slack.append((bus_name, i + 1 + ((day - 1) * 24), slack_pred[0, z, i]))
    
    # Store voltage angle results
    for z in range(n_buses):
        bus_name = list(instance.buses)[z]
        for i in range(n_hours):
            vlt_angle.append((bus_name, i + 1 + ((day - 1) * 24), theta_pred[0, z, i]))
    
    # Store storage results
    for j in range(n_storage):
        storage_name = list(instance.Storage)[j]
        for i in range(n_hours):
            charge.append((storage_name, i + 1 + ((day - 1) * 24), charge_pred[0, j, i]))
            discharge.append((storage_name, i + 1 + ((day - 1) * 24), discharge_pred[0, j, i]))
            SoC.append((storage_name, i + 1 + ((day - 1) * 24), soc_pred[0, j, i + 1]))


def save_final_results(config, mwh, flow, slack, vlt_angle, charge, discharge, SoC, pinn_model=None, system_data=None):
    """Save the final results to files and optionally the trained model"""
    # Convert results to DataFrames
    vlt_angle_pd = pd.DataFrame(vlt_angle, columns=('Node', 'Time', 'Value'))
    mwh_pd = pd.DataFrame(mwh, columns=('Generator', 'Type', 'Time', 'Value'))
    slack_pd = pd.DataFrame(slack, columns=('Node','Time','Value'))
    flow_pd = pd.DataFrame(flow, columns=('Line', 'Time', 'Value'))
    SoC_pd = pd.DataFrame(SoC, columns=('Storage','Time','Value'))
    discharge_pd = pd.DataFrame(discharge, columns=('Storage','Time','Value'))
    charge_pd = pd.DataFrame(charge, columns=('Storage','Time','Value'))
    
    # Save outputs
    vlt_angle_pd.to_parquet(config.vlt_angle_file, index=False)
    mwh_pd.to_parquet(config.mwh_file, index=False)
    slack_pd.to_parquet(config.slack_file, index=False)
    flow_pd.to_parquet(config.flow_file, index=False)
    SoC_pd.to_parquet(config.SoC_file, index=False)
    discharge_pd.to_parquet(config.discharge_file, index=False)
    charge_pd.to_parquet(config.charge_file, index=False)
    
    # Save the trained model if provided
    if pinn_model is not None:
        # Create model directory based on output directory
        model_dir = os.path.dirname(config.mwh_file).replace('mwh', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state dict
        model_file = os.path.join(model_dir, 'pinn_model.pth')
        torch.save(pinn_model.state_dict(), model_file)
        
        # Save model configuration and system data
        config_file = os.path.join(model_dir, 'pinn_config.pkl')
        config_data = {
            'system_data': system_data,
            'model_config': {
                'input_dim': pinn_model.input_dim,
                'output_dim': pinn_model.output_dim,
                'hidden_layers': pinn_model.config.hidden_layers,
                'n_generators': pinn_model.n_generators,
                'n_lines': pinn_model.n_lines,
                'n_buses': pinn_model.n_buses,
                'n_storage': pinn_model.n_storage,
                'n_hours': pinn_model.n_hours,
            }
        }
        with open(config_file, 'wb') as f:
            cloudpickle.dump(config_data, f)
        
        logger.info(f"Trained PINN model saved to: {model_dir}")
        return model_dir
    
    return None 


def load_pinn_model(model_dir: str, device: str = 'cpu'):
    """
    Load a saved PINN model for inference
    
    :param model_dir: Directory containing the saved model files
    :type model_dir: str
    
    :param device: Device to load the model on ('cpu' or 'cuda')
    :type device: str
    
    :return: Tuple of (model, system_data, model_config)
    :rtype: tuple
    """
    import os
    
    # Load model configuration
    config_file = os.path.join(model_dir, 'pinn_config.pkl')
    with open(config_file, 'rb') as f:
        config_data = cloudpickle.load(f)
    
    system_data = config_data['system_data']
    model_config = config_data['model_config']
    
    # Create model instance
    pinn_model = PowerSystemPINN(
        input_dim=model_config['input_dim'],
        output_dim=model_config['output_dim'],
        config=PINNConfig(hidden_layers=model_config['hidden_layers'])
    )
    
    # Set system dimensions
    pinn_model.set_system_dimensions(
        model_config['n_generators'],
        model_config['n_lines'],
        model_config['n_buses'],
        model_config['n_storage'],
        model_config['n_hours']
    )
    
    # Load model weights
    model_file = os.path.join(model_dir, 'pinn_model.pth')
    pinn_model.load_state_dict(torch.load(model_file, map_location=device))
    pinn_model.to(device)
    pinn_model.eval()
    
    logger.info(f"PINN model loaded from: {model_dir}")
    
    return pinn_model, system_data, model_config


def predict_with_saved_model(model_dir: str, inputs: torch.Tensor, device: str = 'cpu'):
    """
    Make predictions using a saved PINN model
    
    :param model_dir: Directory containing the saved model files
    :type model_dir: str
    
    :param inputs: Input tensor (demand data)
    :type inputs: torch.Tensor
    
    :param device: Device to run inference on ('cpu' or 'cuda')
    :type device: str
    
    :return: Model predictions
    :rtype: Dict[str, torch.Tensor]
    """
    # Load model
    model, system_data, model_config = load_pinn_model(model_dir, device)
    
    # Move inputs to device
    inputs = inputs.to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model.predict_variables(inputs)
    
    return predictions 