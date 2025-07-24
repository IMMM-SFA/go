#!/usr/bin/env python3
"""
Test script for the PINN implementation.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import the PINN classes
from go.west.pinn import PINNConfig, PowerSystemPINN, PhysicsLoss, PINNTrainer


class TestPINNConfig(unittest.TestCase):
    """Test the PINN configuration class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PINNConfig()
        self.assertEqual(config.hidden_layers, [256, 256, 256, 256])
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.max_epochs, 10000)
        self.assertEqual(config.device, 'cpu')
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PINNConfig(
            hidden_layers=[128, 64],
            learning_rate=1e-4,
            max_epochs=5000,
            device='cuda'
        )
        self.assertEqual(config.hidden_layers, [128, 64])
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.max_epochs, 5000)
        self.assertEqual(config.device, 'cuda')


class TestPowerSystemPINN(unittest.TestCase):
    """Test the PowerSystemPINN class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PINNConfig(hidden_layers=[64, 32])
        self.input_dim = 100
        self.output_dim = 200
        self.model = PowerSystemPINN(self.input_dim, self.output_dim, self.config)
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsInstance(self.model, PowerSystemPINN)
        self.assertEqual(self.model.config, self.config)
    
    def test_forward_pass(self):
        """Test forward pass."""
        x = torch.randn(10, self.input_dim)
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10, self.output_dim))
    
    def test_set_system_dimensions(self):
        """Test setting system dimensions."""
        self.model.set_system_dimensions(10, 5, 8, 3, 24)
        self.assertEqual(self.model.n_generators, 10)
        self.assertEqual(self.model.n_lines, 5)
        self.assertEqual(self.model.n_buses, 8)
        self.assertEqual(self.model.n_storage, 3)
        self.assertEqual(self.model.n_hours, 24)
    
    def test_predict_variables(self):
        """Test variable prediction."""
        self.model.set_system_dimensions(2, 1, 2, 1, 3)
        inputs = torch.randn(5, 6)  # 2 buses * 3 hours = 6 inputs
        
        predictions = self.model.predict_variables(inputs)
        
        # Check that all expected keys are present
        expected_keys = ['mwh', 'flow', 'theta', 'slack', 'charge', 'discharge', 'soc']
        for key in expected_keys:
            self.assertIn(key, predictions)
        
        # Check output shapes
        self.assertEqual(predictions['mwh'].shape, (5, 6))  # 2 generators * 3 hours
        self.assertEqual(predictions['flow'].shape, (5, 3))  # 1 line * 3 hours
        self.assertEqual(predictions['theta'].shape, (5, 6))  # 2 buses * 3 hours
        self.assertEqual(predictions['slack'].shape, (5, 6))  # 2 buses * 3 hours
        self.assertEqual(predictions['charge'].shape, (5, 3))  # 1 storage * 3 hours
        self.assertEqual(predictions['discharge'].shape, (5, 3))  # 1 storage * 3 hours
        self.assertEqual(predictions['soc'].shape, (5, 4))  # 1 storage * (3+1) hours


class TestPhysicsLoss(unittest.TestCase):
    """Test the PhysicsLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system_data = {
            'n_generators': 2,
            'n_lines': 1,
            'n_buses': 2,
            'n_storage': 1,
            'n_hours': 3,
            'line_to_bus_map': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            'bus_to_unit_map': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            'bus_to_storage_map': torch.tensor([[1.0, 0.0]]),
            'reactance': torch.tensor([0.1]),
            'line_from_bus': [0],
            'line_to_bus': [1],
            'charge_rate': torch.tensor([100.0]),
            'discharge_rate': torch.tensor([100.0]),
            'max_soc': torch.tensor([1000.0]),
            'min_soc': torch.tensor([0.0]),
            'charge_eff': torch.tensor([0.9]),
            'discharge_eff': torch.tensor([0.9])
        }
        self.physics_loss = PhysicsLoss(self.system_data)
    
    def test_nodal_balance_loss(self):
        """Test nodal balance loss calculation."""
        predictions = {
            'mwh': torch.randn(5, 6),  # 2 generators * 3 hours
            'flow': torch.randn(5, 3),  # 1 line * 3 hours
            'slack': torch.randn(5, 6),  # 2 buses * 3 hours
            'charge': torch.randn(5, 3),  # 1 storage * 3 hours
            'discharge': torch.randn(5, 3)  # 1 storage * 3 hours
        }
        demand = torch.randn(2, 3)  # 2 buses * 3 hours
        must_run = torch.randn(2, 3)  # 2 buses * 3 hours
        
        loss = self.physics_loss.nodal_balance_loss(predictions, demand, must_run)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_capacity_constraints_loss(self):
        """Test capacity constraints loss calculation."""
        predictions = {
            'mwh': torch.randn(5, 6)  # 2 generators * 3 hours
        }
        max_capacity = torch.ones(2, 3) * 100  # 2 generators * 3 hours
        
        loss = self.physics_loss.capacity_constraints_loss(predictions, max_capacity)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_storage_constraints_loss(self):
        """Test storage constraints loss calculation."""
        predictions = {
            'charge': torch.randn(5, 3),  # 1 storage * 3 hours
            'discharge': torch.randn(5, 3),  # 1 storage * 3 hours
            'soc': torch.randn(5, 4)  # 1 storage * (3+1) hours
        }
        
        loss = self.physics_loss.storage_constraints_loss(predictions)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_objective_loss(self):
        """Test objective loss calculation."""
        predictions = {
            'mwh': torch.randn(5, 6),  # 2 generators * 3 hours
            'slack': torch.randn(5, 6),  # 2 buses * 3 hours
            'flow': torch.randn(5, 3),  # 1 line * 3 hours
            'charge': torch.randn(5, 3),  # 1 storage * 3 hours
            'discharge': torch.randn(5, 3)  # 1 storage * 3 hours
        }
        fuel_prices = torch.randn(2)  # 2 thermal generators
        heat_rates = torch.randn(2)  # 2 generators
        
        loss = self.physics_loss.objective_loss(predictions, fuel_prices, heat_rates)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)


class TestPINNTrainer(unittest.TestCase):
    """Test the PINNTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PINNConfig(hidden_layers=[32, 16])
        self.model = PowerSystemPINN(10, 20, self.config)
        self.model.set_system_dimensions(2, 1, 2, 1, 3)
        
        system_data = {
            'n_generators': 2,
            'n_lines': 1,
            'n_buses': 2,
            'n_storage': 1,
            'n_hours': 3,
            'line_to_bus_map': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            'bus_to_unit_map': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            'bus_to_storage_map': torch.tensor([[1.0, 0.0]]),
            'reactance': torch.tensor([0.1]),
            'line_from_bus': [0],
            'line_to_bus': [1],
            'charge_rate': torch.tensor([100.0]),
            'discharge_rate': torch.tensor([100.0]),
            'max_soc': torch.tensor([1000.0]),
            'min_soc': torch.tensor([0.0]),
            'charge_eff': torch.tensor([0.9]),
            'discharge_eff': torch.tensor([0.9])
        }
        self.physics_loss = PhysicsLoss(system_data)
        self.trainer = PINNTrainer(self.model, self.physics_loss, self.config)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        self.assertIsInstance(self.trainer, PINNTrainer)
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.physics_loss, self.physics_loss)
        self.assertEqual(self.trainer.config, self.config)
    
    def test_train_step(self):
        """Test single training step."""
        inputs = torch.randn(5, 10)  # 5 samples, 10 input features
        targets = {
            'demand': torch.randn(2, 3),  # 2 buses * 3 hours
            'must_run': torch.randn(2, 3),  # 2 buses * 3 hours
            'max_capacity': torch.ones(2, 3) * 100,  # 2 generators * 3 hours
            'fuel_prices': torch.randn(2),  # 2 thermal generators
            'heat_rates': torch.randn(2)  # 2 generators
        }
        
        losses = self.trainer.train_step(inputs, targets)
        
        self.assertIn('total_loss', losses)
        self.assertIn('physics_loss', losses)
        self.assertIn('objective_loss', losses)
        
        for loss_value in losses.values():
            self.assertIsInstance(loss_value, float)
            self.assertGreater(loss_value, 0)


if __name__ == '__main__':
    unittest.main() 