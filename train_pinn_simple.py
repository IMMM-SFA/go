#!/usr/bin/env python3
"""
Simple PINN Training Script for IM3 Data
"""

import sys
import os
from pathlib import Path
import logging
import yaml
import re

# Add the go package to the path
sys.path.insert(0, str(Path(__file__).parent))

from go.west.pinn import west_linear_multi_pinn, PINNConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_years_from_path(path):
    # Extract parent year from directory
    parent_year_match = re.search(r'/(\d{4})/', path)
    parent_year = int(parent_year_match.group(1)) if parent_year_match else -1

    # Extract start year from filename (first 4-digit number after 'go_config_')
    filename = os.path.basename(path)
    start_year_match = re.search(r'go_config_(\d{4})', filename)
    start_year = int(start_year_match.group(1)) if start_year_match else -1

    return (parent_year, start_year)


def main():
    """Main training function"""
    
    # Load configuration
    with open("pinn_training_config.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    # PINN Configuration
    pinn_config = PINNConfig(
        hidden_layers=config['pinn']['hidden_layers'],
        learning_rate=config['pinn']['learning_rate'],
        max_epochs=config['pinn']['max_epochs'],
        patience=config['pinn']['patience'],
        physics_weight=config['pinn']['physics_weight'],
        objective_weight=config['pinn']['objective_weight'],
        constraint_weight=config['pinn']['constraint_weight'],
        batch_size=config['pinn']['batch_size'],
        device=config['pinn']['device']
    )
    
    # Training parameters
    n_days = config['training']['n_days']
    save_restart_file = config['training']['save_restart_file']
    restart_write_frequency = config['training']['restart_write_frequency']
    
    # Data directory
    corrected_configs_dir = config['data']['corrected_configs_dir']
    
    logger.info("Starting PINN training for IM3 experiment data")
    logger.info(f"Corrected configs directory: {corrected_configs_dir}")
    
    # Find all corrected config files
    config_files = []
    for root, dirs, files in os.walk(corrected_configs_dir):
        for file in files:
            if file.endswith('.yml'):
                config_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(config_files)} configuration files")

    # Sort config files by "year order" as described:
    # - Extract the "year" from the parent directory (e.g., .../2020/)
    # - Extract the "start year" from the filename (e.g., go_config_2015PI_2015RG_2015TC_2020LD.yml -> 2015)
    # - Sort first by parent year, then by start year
    config_files.sort(key=extract_years_from_path)

    # Training loop
    for i, config_file in enumerate(config_files[:2]):  # Start with first 2 for testing
        logger.info(f"Training {i+1}/{len(config_files)}: {config_file}")
        
        try:
            # Run PINN training
            result = west_linear_multi_pinn(
                config_file=config_file,
                pinn_config=pinn_config,
                n_days=n_days,
                save_restart_file=save_restart_file,
                restart_write_frequency=restart_write_frequency
            )
            
            logger.info(f"Completed training for {config_file}")
        
        except Exception as e:
            logger.error(f"Error training {config_file}: {e}")
            continue
    
    logger.info("PINN training completed")

if __name__ == "__main__":
    main()
