#!/usr/bin/env python3
"""
Example script demonstrating the use of the Physics Informed Neural Network (PINN)
version of the west_linear_multi model.

This example shows how to:
1. Configure the PINN model
2. Run a simulation using the PINN approach
3. Compare results with traditional optimization
"""

import os
import sys
import logging
from pathlib import Path

# Add the go package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from go.west.pinn import west_linear_multi_pinn, PINNConfig
from go.west.launch import west_linear_multi


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pinn_example.log'),
            logging.StreamHandler()
        ]
    )


def run_pinn_example():
    """Run the PINN example"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting PINN example")
    
    # Configuration file path (adjust this to your actual config file)
    config_file = "path/to/your/config.yml"  # Update this path
    
    # PINN Configuration
    pinn_config = PINNConfig(
        hidden_layers=[128, 128, 128],  # Smaller network for faster training
        learning_rate=1e-3,
        max_epochs=5000,  # Fewer epochs for demonstration
        patience=500,
        physics_weight=1.0,
        objective_weight=1.0,
        constraint_weight=10.0,
        batch_size=16,
        device='cpu'  # Use 'cuda' if GPU is available
    )
    
    try:
        # Run PINN simulation
        logger.info("Running PINN simulation...")
        pinn_result = west_linear_multi_pinn(
            config_file=config_file,
            pinn_config=pinn_config,
            n_days=7,  # Run for 1 week
            save_restart_file=True,
            restart_write_frequency=3
        )
        
        logger.info(f"PINN simulation completed. Processed {pinn_result} days.")
        
        # Optionally run traditional optimization for comparison
        logger.info("Running traditional optimization for comparison...")
        traditional_result = west_linear_multi(
            config_file=config_file,
            solver_name="appsi_highs",
            n_days=7,
            save_restart_file=True,
            restart_write_frequency=3
        )
        
        logger.info(f"Traditional optimization completed. Processed {traditional_result} days.")
        
        # Compare results (this would need to be implemented based on your specific needs)
        compare_results()
        
    except Exception as e:
        logger.error(f"Error running PINN example: {e}")
        raise


def compare_results():
    """Compare PINN results with traditional optimization results"""
    logger = logging.getLogger(__name__)
    logger.info("Comparing PINN and traditional optimization results...")
    
    # This function would implement the comparison logic
    # For example:
    # 1. Load results from both approaches
    # 2. Calculate objective function values
    # 3. Check constraint violations
    # 4. Compare solution times
    # 5. Generate comparison plots
    
    logger.info("Results comparison completed.")


def run_parameter_study():
    """Run a parameter study to find optimal PINN configuration"""
    logger = logging.getLogger(__name__)
    logger.info("Running PINN parameter study...")
    
    # Test different network architectures
    architectures = [
        [64, 64],
        [128, 128],
        [256, 256],
        [128, 128, 128],
        [256, 256, 256]
    ]
    
    # Test different learning rates
    learning_rates = [1e-2, 1e-3, 1e-4]
    
    # Test different physics weights
    physics_weights = [0.1, 1.0, 10.0]
    
    best_config = None
    best_loss = float('inf')
    
    for arch in architectures:
        for lr in learning_rates:
            for pw in physics_weights:
                logger.info(f"Testing architecture: {arch}, lr: {lr}, physics_weight: {pw}")
                
                pinn_config = PINNConfig(
                    hidden_layers=arch,
                    learning_rate=lr,
                    physics_weight=pw,
                    max_epochs=1000,  # Shorter training for parameter study
                    patience=200
                )
                
                try:
                    # Run a short simulation to test the configuration
                    result = west_linear_multi_pinn(
                        config_file="path/to/your/config.yml",  # Update this path
                        pinn_config=pinn_config,
                        n_days=1,  # Just one day for parameter study
                        save_restart_file=False
                    )
                    
                    # Here you would evaluate the result and update best_config
                    # This is a placeholder for the actual evaluation logic
                    
                except Exception as e:
                    logger.warning(f"Configuration failed: {e}")
                    continue
    
    logger.info(f"Parameter study completed. Best configuration: {best_config}")


def main():
    """Main function"""
    print("PINN Example for West Linear Multi Model")
    print("=" * 50)
    
    # Check if config file exists
    config_file = "path/to/your/config.yml"  # Update this path
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        print("Please update the config_file path in the script.")
        return
    
    # Run the example
    run_pinn_example()
    
    # Optionally run parameter study
    # run_parameter_study()


if __name__ == "__main__":
    main() 