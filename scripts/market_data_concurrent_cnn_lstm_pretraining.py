#!/usr/bin/env python3
"""
Concurrent Returns CNN-LSTM Pretraining Script

This script trains a CNN-LSTM model on concurrent/past returns instead of future returns.
It simply imports and calls the main pretraining functions from the original script with
past_returns=True to enable concurrent returns functionality.

This approach eliminates code duplication while providing a clean interface for
concurrent returns experiments.
"""

# Import the main function from the original pretraining script
from market_data_cnn_lstm_pretraining import main as main_pretraining

def main():
    """
    Main function to train CNN-LSTM model for concurrent returns.
    
    This function simply calls the original main() function with past_returns=True
    to enable concurrent returns training instead of future returns.
    """
    return main_pretraining(num_epochs=1, past_returns=True)

if __name__ == "__main__":
    main()
