#!/usr/bin/env python3
"""
PINN Validation Script for IM3 Experiment Data
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Add the go package to the path
sys.path.insert(0, str(Path(__file__).parent))

from go.west.pinn import west_linear_multi_pinn, PINNConfig
from go.west.launch import west_linear_multi

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_results(pinn_file: str, traditional_file: str) -> dict:
    """Compare PINN results with traditional optimization results"""
    
    try:
        # Load results
        pinn_results = pd.read_parquet(pinn_file)
        traditional_results = pd.read_parquet(traditional_file)
        
        # Calculate metrics
        metrics = {}
        
        # For MWH files
        if 'Generator' in pinn_results.columns:
            pinn_total = pinn_results['Value'].sum()
            traditional_total = traditional_results['Value'].sum()
            metrics['total_generation'] = {
                'pinn': float(pinn_total),
                'traditional': float(traditional_total),
                'difference_pct': float(((pinn_total - traditional_total) / traditional_total) * 100)
            }
        
        # For flow files
        if 'Line' in pinn_results.columns:
            pinn_flow = pinn_results['Value'].abs().mean()
            traditional_flow = traditional_results['Value'].abs().mean()
            metrics['avg_flow'] = {
                'pinn': float(pinn_flow),
                'traditional': float(traditional_flow),
                'difference_pct': float(((pinn_flow - traditional_flow) / traditional_flow) * 100)
            }
        
        return metrics
    except Exception as e:
        logger.error(f"Error comparing results: {e}")
        return {}

def main():
    """Main validation function"""
    
    base_path = "/Users/d3y010/projects/go/data/rcfs/projects/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/run_110824/rcp45cooler_ssp3/go"
    validation_years = [2045]
    
    logger.info("Starting PINN validation")
    
    # PINN configuration for validation
    pinn_config = PINNConfig(
        hidden_layers=[256, 256, 256, 256],
        learning_rate=1e-4,  # Lower learning rate for validation
        max_epochs=2000,
        patience=500,
        physics_weight=10.0,  # Emphasize constraint satisfaction
        objective_weight=1.0,
        constraint_weight=10.0,
        batch_size=16,
        device='cpu'
    )
    
    results = []
    
    for year in validation_years:
        logger.info(f"Validating year {year}")
        
        config_dir = Path(base_path) / "input" / "go_config_yml" / str(year)
        if not config_dir.exists():
            continue
        
        for config_file in config_dir.glob("*.yml"):
            scenario = config_file.stem.replace("go_config_", "")
            
            try:
                # Run PINN
                pinn_result = west_linear_multi_pinn(
                    config_file=str(config_file),
                    pinn_config=pinn_config,
                    n_days=7,  # One week for validation
                    save_restart_file=False
                )
                
                # Run traditional optimization for comparison
                traditional_result = west_linear_multi(
                    config_file=str(config_file),
                    solver_name="appsi_highs",
                    n_days=7
                )
                
                # Compare results
                pinn_output_file = f"{base_path}/output/native_output/{year}/mwh/mwh_{scenario}.parquet"
                traditional_output_file = f"{base_path}/output/native_output/{year}/mwh/mwh_{scenario}.parquet"
                
                if os.path.exists(pinn_output_file) and os.path.exists(traditional_output_file):
                    metrics = compare_results(pinn_output_file, traditional_output_file)
                    
                    results.append({
                        'year': year,
                        'scenario': scenario,
                        'metrics': metrics
                    })
                
                logger.info(f"Validation completed for {scenario} in {year}")
                
            except Exception as e:
                logger.error(f"Error validating {scenario} in {year}: {e}")
                continue
    
    # Print summary
    logger.info("Validation Summary:")
    for result in results:
        logger.info(f"{result['year']} - {result['scenario']}: {result['metrics']}")

if __name__ == "__main__":
    main()
