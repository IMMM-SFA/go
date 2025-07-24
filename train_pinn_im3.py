#!/usr/bin/env python3
"""
PINN Training Script for IM3 Experiment Data
Generated automatically based on data analysis
"""

import sys
import os
from pathlib import Path
import logging
import torch
import pandas as pd
import numpy as np

# Add the go package to the path
sys.path.insert(0, str(Path(__file__).parent))

from go.west.pinn import west_linear_multi_pinn, PINNConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pinn_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    
    # PINN Configuration
    pinn_config = PINNConfig(
        hidden_layers=[256, 256, 256, 256],
        learning_rate=0.001,
        max_epochs=8000,
        patience=1000,
        physics_weight=1.0,
        objective_weight=1.0,
        constraint_weight=10.0,
        batch_size=16,
        device='cpu'
    )
    
    # Training configuration
    base_path = "/Users/d3y010/projects/go/data/rcfs/projects/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/run_110824/rcp45cooler_ssp3/go"
    training_years = [2020, 2025, 2030, 2035, 2040]
    
    logger.info("Starting PINN training for IM3 experiment data")
    logger.info(f"Training years: {training_years}")
    
    # Training loop
    for year in training_years:
        logger.info(f"Training for year {year}")
        
        # Find config files for this year
        config_dir = Path(base_path) / "input" / "go_config_yml" / str(year)
        if not config_dir.exists():
            logger.warning(f"No config directory for year {year}")
            continue
        
        for config_file in config_dir.glob("*.yml"):
            scenario = config_file.stem.replace("go_config_", "")
            logger.info(f"Training scenario: {scenario}")
            
            try:
                # Run PINN training
                result = west_linear_multi_pinn(
                    config_file=str(config_file),
                    pinn_config=pinn_config,
                    n_days=30,  # Start with one month for testing
                    save_restart_file=True,
                    restart_write_frequency=7  # Save every week
                )
                
                logger.info(f"Completed training for {scenario} in {year}")
                
            except Exception as e:
                logger.error(f"Error training {scenario} in {year}: {e}")
                continue
    
    logger.info("PINN training completed")

if __name__ == "__main__":
    main()
