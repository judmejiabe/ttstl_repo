# TTSTL Usage Guide

## Installation

### Basic Installation
```bash
pip install ttstl
```

### Development Installation
```bash
git clone https://github.com/judmejiabe/ttstl_repo.git
cd ttstl_repo
pip install -e .
```

## Quick Start

### 1. Basic TTSTL Usage

```python
import ttstl
import pandas as pd
import numpy as np

# Load your time series data
data = pd.read_csv('your_data.csv')

# Initialize TTSTL model
from ttstl import TTSTLOrchestrator, CNNLSTM, Interface, Adapter

# Create components
backbone = CNNLSTM(input_dim=10, hidden_dim=64)
interface = Interface(detection_method='cusum')
adapter = Adapter(model_type='rugarch')

# Create orchestrator
ttstl_model = TTSTLOrchestrator(
    backbone=backbone,
    interface=interface,
    adapter=adapter
)

# Train on historical data
ttstl_model.global_retrain(train_data)

# Make predictions
predictions = ttstl_model.predict(test_data)
```

### 2. Working with Financial Data

```python
from ttstl.utils import preprocess_mbo_data_cnn_lstm

# Preprocess limit order book data
features, targets = preprocess_mbo_data_cnn_lstm(
    path='lob_data.parquet',
    delta=0.1,  # 100ms sampling
    quantile=0.005
)

# Split data
train_size = int(0.8 * len(features))
train_features = features[:train_size]
train_targets = targets[:train_size]
test_features = features[train_size:]
test_targets = targets[train_size:]

# Train model
ttstl_model.global_retrain(train_features, train_targets)

# Adaptive prediction loop
for i in range(len(test_features)):
    # Check if adaptation is needed
    if ttstl_model.interface.detect_change(test_features[i]):
        ttstl_model.local_adapt(test_features[i])
    
    # Make prediction
    pred = ttstl_model.predict(test_features[i])
    print(f"Prediction: {pred}")
```

## Command Line Interface

### Global Retraining

```bash
# Run global retraining
ttstl-global-retrain \
    --data-path data/train.csv \
    --config-path configs/backbone_config.yaml \
    --output-path models/backbone_model.pth
```

### Local Adaptation

```bash
# Run local adaptation loop
ttstl-local-adapt \
    --data-path data/stream.csv \
    --model-path models/backbone_model.pth \
    --config-path configs/adapter_config.yaml \
    --output-path results/
```

### Forecasting

```bash
# Generate forecasts
ttstl-forecast \
    --data-path data/new_data.csv \
    --model-path models/ttstl_model.pkl \
    --horizon 10 \
    --output-path forecasts.csv
```

## Configuration

### Backbone Configuration

```yaml
# configs/backbone_config.yaml
backbone:
  model_type: "cnn_lstm"
  input_dim: 10
  hidden_dim: 64
  num_layers: 2
  dropout: 0.1
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

data:
  sequence_length: 50
  target_horizons: [1, 5, 10]
  validation_split: 0.2
```

### Interface Configuration

```yaml
# configs/interface_config.yaml
interface:
  detection_method: "cusum"
  threshold: 0.05
  window_size: 100
  warmup_period: 50
  
  # Alternative: statistical tests
  # detection_method: "ks_test"
  # significance_level: 0.01
```

### Adapter Configuration

```yaml
# configs/adapter_config.yaml
adapter:
  model_type: "rugarch"
  garch_spec: "sGARCH"
  distribution: "std"
  mean_model: "ARFIMA"
  update_frequency: 10
  window_size: 500
```

## Advanced Usage

### Custom Backbone

```python
from ttstl.backbone import Backbone

class CustomBackbone(Backbone):
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize your custom model
        
    def forward(self, x):
        # Implement forward pass
        pass
    
    def fit(self, data, targets):
        # Implement training
        pass

# Use custom backbone
custom_backbone = CustomBackbone(custom_param=value)
ttstl_model = TTSTLOrchestrator(
    backbone=custom_backbone,
    interface=interface,
    adapter=adapter
)
```

### Custom Change Detection

```python
from ttstl.interface import Interface

class CustomInterface(Interface):
    def detect_change(self, data):
        # Implement custom change detection
        return change_detected
    
    def get_adaptation_signal(self, data):
        # Return adaptation strength
        return signal_strength
```

### Batch Processing

```python
# Process multiple time series
results = []
for series_id, series_data in data_dict.items():
    # Initialize fresh model for each series
    model = TTSTLOrchestrator(backbone, interface, adapter)
    
    # Train and predict
    model.global_retrain(series_data['train'])
    predictions = model.predict(series_data['test'])
    
    results.append({
        'series_id': series_id,
        'predictions': predictions
    })
```

## Best Practices

### 1. Data Preparation
- Ensure data is properly normalized
- Handle missing values appropriately
- Use appropriate sampling frequencies

### 2. Model Selection
- Start with default configurations
- Tune hyperparameters on validation set
- Consider domain-specific requirements

### 3. Monitoring
- Track adaptation frequency
- Monitor prediction performance
- Log change detection events

### 4. Deployment
- Use proper versioning for models
- Implement proper error handling
- Monitor system performance in production
