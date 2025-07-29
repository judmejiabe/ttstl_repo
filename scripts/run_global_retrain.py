#!/usr/bin/env python3
"""
Global Retraining Script for TTSTL

This script performs a full global retrain of the TTSTL backbone model
on historical data.
"""

import argparse
import logging
import yaml
import pandas as pd
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ttstl.backbone import CNNLSTM
from ttstl.utils import preprocess_mbo_data_cnn_lstm, get_datasets_cnn_lstm

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('global_retrain.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='TTSTL Global Retraining')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the training data file')
    parser.add_argument('--config-path', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the trained model')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting TTSTL global retraining...")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config_path}")
        config = load_config(args.config_path)
        
        # Load and preprocess data
        logger.info(f"Loading data from {args.data_path}")
        features, targets = preprocess_mbo_data_cnn_lstm(
            path=args.data_path,
            delta=config.get('data', {}).get('delta', 0.1),
            quantile=config.get('data', {}).get('quantile', 0.005)
        )
        
        logger.info(f"Data loaded: features={features.shape}, targets={targets.shape}")
        
        # Create datasets
        train_loader, val_loader = get_datasets_cnn_lstm(
            features=features,
            targets=targets,
            sequence_length=config.get('data', {}).get('sequence_length', 50),
            batch_size=config.get('backbone', {}).get('batch_size', 32),
            train_split=config.get('data', {}).get('train_split', 0.8)
        )
        
        # Initialize model
        backbone_config = config.get('backbone', {})
        model = CNNLSTM(
            input_dim=backbone_config.get('input_dim', features.shape[1]),
            hidden_dim=backbone_config.get('hidden_dim', 64),
            output_dim=targets.shape[1],
            **{k: v for k, v in backbone_config.items() 
               if k not in ['input_dim', 'hidden_dim']}
        )
        
        logger.info(f"Model initialized: {model}")
        
        # Train model
        logger.info("Starting training...")
        training_history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=backbone_config.get('epochs', 100),
            learning_rate=backbone_config.get('learning_rate', 0.001)
        )
        
        logger.info("Training completed successfully!")
        
        # Save model
        logger.info(f"Saving model to {args.output_path}")
        model.save(args.output_path)
        
        # Save training history
        history_path = Path(args.output_path).parent / "training_history.yaml"
        with open(history_path, 'w') as f:
            yaml.dump(training_history, f)
        
        logger.info(f"Global retraining completed successfully!")
        logger.info(f"Model saved to: {args.output_path}")
        logger.info(f"Training history saved to: {history_path}")
        
    except Exception as e:
        logger.error(f"Error during global retraining: {e}")
        raise

if __name__ == "__main__":
    main()
