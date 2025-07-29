#!/usr/bin/env python3
"""
Local Adaptation Script for TTSTL

This script runs the local adaptation loop for TTSTL, monitoring incoming
data for distribution changes and triggering local model updates.
"""

import argparse
import logging
import yaml
import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ttstl.orchestrator import TTSTLOrchestrator
from ttstl.backbone import CNNLSTM
from ttstl.interface import Interface
from ttstl.adapter import Adapter

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('local_adapt.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_streaming_data(data_path, chunk_size=1000):
    """
    Generator that yields chunks of streaming data
    """
    data = pd.read_csv(data_path)
    
    for i in range(0, len(data), chunk_size):
        yield data.iloc[i:i+chunk_size]

def main():
    parser = argparse.ArgumentParser(description='TTSTL Local Adaptation')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the streaming data file')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the pretrained backbone model')
    parser.add_argument('--config-path', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save adaptation results')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Size of data chunks to process')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting TTSTL local adaptation...")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config_path}")
        config = load_config(args.config_path)
        
        # Load pretrained backbone model
        logger.info(f"Loading pretrained model from {args.model_path}")
        backbone = CNNLSTM.load(args.model_path)
        
        # Initialize interface and adapter
        interface_config = config.get('interface', {})
        interface = Interface(**interface_config)
        
        adapter_config = config.get('adapter', {})
        adapter = Adapter(**adapter_config)
        
        # Create TTSTL orchestrator
        ttstl_model = TTSTLOrchestrator(
            backbone=backbone,
            interface=interface,
            adapter=adapter
        )
        
        logger.info("TTSTL orchestrator initialized successfully")
        
        # Initialize tracking variables
        adaptation_events = []
        predictions = []
        performance_metrics = []
        
        # Start local adaptation loop
        logger.info("Starting local adaptation loop...")
        
        chunk_count = 0
        for data_chunk in load_streaming_data(args.data_path, args.chunk_size):
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count} with {len(data_chunk)} samples")
            
            # Check for distribution changes
            change_detected = interface.detect_change(data_chunk)
            adaptation_signal = interface.get_adaptation_signal(data_chunk)
            
            if change_detected:
                logger.info(f"Distribution change detected! Signal strength: {adaptation_signal:.4f}")
                
                # Perform local adaptation
                adaptation_result = ttstl_model.local_adapt(data_chunk)
                adaptation_events.append({
                    'chunk': chunk_count,
                    'timestamp': time.time(),
                    'signal_strength': adaptation_signal,
                    'adaptation_result': adaptation_result
                })
                
                logger.info(f"Local adaptation completed for chunk {chunk_count}")
            
            # Generate predictions
            chunk_predictions = ttstl_model.predict(data_chunk)
            predictions.extend(chunk_predictions)
            
            # Calculate performance metrics (if targets available)
            if 'target' in data_chunk.columns:
                # Calculate prediction error
                error = np.mean(np.abs(chunk_predictions - data_chunk['target'].values))
                performance_metrics.append({
                    'chunk': chunk_count,
                    'mae': error,
                    'adaptation_signal': adaptation_signal
                })
                logger.info(f"Chunk {chunk_count} MAE: {error:.6f}")
            
            # Update reference distribution for interface
            interface.update_reference(data_chunk)
            
            # Periodic logging
            if chunk_count % 10 == 0:
                logger.info(f"Processed {chunk_count} chunks, "
                           f"{len(adaptation_events)} adaptations triggered")
        
        # Save results
        logger.info(f"Saving results to {args.output_path}")
        
        results = {
            'adaptation_events': adaptation_events,
            'predictions': predictions,
            'performance_metrics': performance_metrics,
            'config': config,
            'total_chunks': chunk_count
        }
        
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save adaptation events
        pd.DataFrame(adaptation_events).to_csv(output_path / "adaptation_events.csv", index=False)
        
        # Save predictions
        pd.DataFrame({'predictions': predictions}).to_csv(output_path / "predictions.csv", index=False)
        
        # Save performance metrics
        if performance_metrics:
            pd.DataFrame(performance_metrics).to_csv(output_path / "performance_metrics.csv", index=False)
        
        # Save summary
        with open(output_path / "summary.yaml", 'w') as f:
            yaml.dump({
                'total_chunks_processed': chunk_count,
                'total_adaptations': len(adaptation_events),
                'adaptation_rate': len(adaptation_events) / chunk_count,
                'final_performance': performance_metrics[-1] if performance_metrics else None
            }, f)
        
        logger.info(f"Local adaptation completed successfully!")
        logger.info(f"Total chunks processed: {chunk_count}")
        logger.info(f"Total adaptations: {len(adaptation_events)}")
        logger.info(f"Adaptation rate: {len(adaptation_events) / chunk_count:.4f}")
        
    except Exception as e:
        logger.error(f"Error during local adaptation: {e}")
        raise

if __name__ == "__main__":
    main()
