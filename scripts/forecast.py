#!/usr/bin/env python3
"""
Forecasting Script for TTSTL

This script generates forecasts using a trained TTSTL model.
"""

import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ttstl.orchestrator import TTSTLOrchestrator

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('forecast.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_data(data_path):
    """Load data from file"""
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

def main():
    parser = argparse.ArgumentParser(description='TTSTL Forecasting')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the input data file')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained TTSTL model')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save forecasts')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Forecasting horizon')
    parser.add_argument('--confidence-intervals', action='store_true',
                        help='Generate confidence intervals')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting TTSTL forecasting...")
    
    try:
        # Load data
        logger.info(f"Loading data from {args.data_path}")
        data = load_data(args.data_path)
        logger.info(f"Data loaded: {data.shape}")
        
        # Load TTSTL model
        logger.info(f"Loading TTSTL model from {args.model_path}")
        ttstl_model = TTSTLOrchestrator.load_model(args.model_path)
        logger.info("TTSTL model loaded successfully")
        
        # Generate forecasts
        logger.info(f"Generating forecasts with horizon {args.horizon}")
        forecasts = ttstl_model.predict(
            data=data,
            horizon=args.horizon
        )
        
        logger.info(f"Generated {len(forecasts)} forecasts")
        
        # Prepare results
        results = pd.DataFrame({
            'forecast': forecasts,
            'timestamp': data['timestamp'].iloc[-len(forecasts):] if 'timestamp' in data.columns else range(len(forecasts))
        })
        
        # Generate confidence intervals if requested
        if args.confidence_intervals:
            logger.info("Generating confidence intervals...")
            
            # Use adapter for uncertainty estimation
            if hasattr(ttstl_model.adapter, 'forecast_with_intervals'):
                intervals = ttstl_model.adapter.forecast_with_intervals(
                    data=data,
                    horizon=args.horizon,
                    confidence_levels=[0.95, 0.68]
                )
                
                results['lower_95'] = intervals['lower_95']
                results['upper_95'] = intervals['upper_95']
                results['lower_68'] = intervals['lower_68']
                results['upper_68'] = intervals['upper_68']
            else:
                # Simple approximation using historical volatility
                recent_volatility = data['micro'].pct_change().rolling(100).std().iloc[-1]
                results['lower_95'] = forecasts - 1.96 * recent_volatility
                results['upper_95'] = forecasts + 1.96 * recent_volatility
                results['lower_68'] = forecasts - 1.0 * recent_volatility
                results['upper_68'] = forecasts + 1.0 * recent_volatility
                
                logger.info("Generated approximate confidence intervals")
        
        # Add metadata
        results_metadata = {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'horizon': args.horizon,
            'n_forecasts': len(forecasts),
            'confidence_intervals': args.confidence_intervals,
            'forecast_mean': float(np.mean(forecasts)),
            'forecast_std': float(np.std(forecasts))
        }
        
        # Save results
        logger.info(f"Saving forecasts to {args.output_path}")
        
        # Ensure output directory exists
        output_path = Path(args.output_path)
        if output_path.suffix in ['.csv', '.parquet']:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save forecasts
            if output_path.suffix == '.csv':
                results.to_csv(output_path, index=False)
            else:
                results.to_parquet(output_path, index=False)
            
            # Save metadata
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.yaml"
            with open(metadata_path, 'w') as f:
                yaml.dump(results_metadata, f)
                
        else:
            # Output path is a directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save forecasts
            results.to_csv(output_path / "forecasts.csv", index=False)
            
            # Save metadata
            with open(output_path / "metadata.yaml", 'w') as f:
                yaml.dump(results_metadata, f)
        
        logger.info("Forecasting completed successfully!")
        logger.info(f"Generated {len(forecasts)} forecasts")
        logger.info(f"Forecast statistics: mean={np.mean(forecasts):.6f}, std={np.std(forecasts):.6f}")
        
        # Basic forecast summary
        print("\nForecast Summary:")
        print("=" * 50)
        print(f"Number of forecasts: {len(forecasts)}")
        print(f"Horizon: {args.horizon}")
        print(f"Mean forecast: {np.mean(forecasts):.6f}")
        print(f"Forecast std: {np.std(forecasts):.6f}")
        print(f"Min forecast: {np.min(forecasts):.6f}")
        print(f"Max forecast: {np.max(forecasts):.6f}")
        
        if args.confidence_intervals:
            print(f"95% CI width (avg): {np.mean(results['upper_95'] - results['lower_95']):.6f}")
        
    except Exception as e:
        logger.error(f"Error during forecasting: {e}")
        raise

if __name__ == "__main__":
    main()
