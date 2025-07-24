#!/usr/bin/env python3
"""
Example: Using a Saved PINN Model for Inference

This script demonstrates how to load a trained PINN model and use it
for making predictions on new data.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import logging

# Add the go package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from go.west.pinn import load_pinn_model, predict_with_saved_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_demand_data(config_file: str, day: int = 1):
    """Load demand data for a specific day"""
    import yaml
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load demand data
    demand_file = config['nodal_load_file']
    demand_data = pd.read_csv(demand_file)
    
    # Extract demand for the specified day (24 hours)
    start_hour = (day - 1) * 24
    end_hour = day * 24
    
    # Assuming first column is time, rest are bus demands
    demand_values = demand_data.iloc[start_hour:end_hour, 1:].values
    
    return demand_values, config


def main():
    """Main function demonstrating PINN model inference"""
    
    # Example model directory (replace with actual path after training)
    model_dir = "/Users/d3y010/projects/go/data/rcfs/projects/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/run_110824/rcp45cooler_ssp3/go/output/native_output/2020/models"
    
    # Example config file (replace with actual path)
    config_file = "/Users/d3y010/projects/go/data/rcfs/projects/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/run_110824/rcp45cooler_ssp3/go/input/go_config_yml_corrected/2020/go_config_2020PI_2020RG_2020TC_2020LD.yml"
    
    # Check if model exists
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        logger.info("Please train a PINN model first using train_pinn_simple.py")
        return
    
    # Load demand data for day 1
    try:
        demand_data, config = load_demand_data(config_file, day=1)
        logger.info(f"Loaded demand data shape: {demand_data.shape}")
    except Exception as e:
        logger.error(f"Error loading demand data: {e}")
        return
    
    # Convert to tensor
    inputs = torch.FloatTensor(demand_data.flatten()).unsqueeze(0)  # Add batch dimension
    logger.info(f"Input tensor shape: {inputs.shape}")
    
    # Make predictions using saved model
    try:
        predictions = predict_with_saved_model(model_dir, inputs, device='cpu')
        logger.info("Successfully made predictions with saved PINN model")
        
        # Print prediction summary
        for key, value in predictions.items():
            logger.info(f"{key}: {value.shape}")
        
        # Example: Extract generation predictions
        if 'mwh' in predictions:
            mwh_pred = predictions['mwh']
            logger.info(f"Generation predictions shape: {mwh_pred.shape}")
            logger.info(f"Total generation: {mwh_pred.sum().item():.2f} MWh")
        
        # Example: Extract flow predictions
        if 'flow' in predictions:
            flow_pred = predictions['flow']
            logger.info(f"Flow predictions shape: {flow_pred.shape}")
            logger.info(f"Average flow: {flow_pred.abs().mean().item():.2f} MW")
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return
    
    logger.info("PINN inference completed successfully!")


if __name__ == "__main__":
    main() 