# TTSTL Architecture

## Overview

The Two Time Scale Transfer Learning (TTSTL) framework is designed with a modular architecture that separates concerns and enables flexible experimentation and deployment.

## Core Components

### 1. Backbone (`backbone.py`)

The backbone component handles the core neural network architecture and feature extraction:

```python
class Backbone:
    """Base class for neural network backbones"""
    
class CNNLSTM(Backbone):
    """CNN-LSTM backbone for time series feature extraction"""
    
class CNNLSTMDataset:
    """Dataset wrapper for CNN-LSTM training"""
```

**Responsibilities:**
- Feature extraction from raw time series data
- Neural network training and inference
- Model checkpointing and loading

### 2. Interface (`interface.py`)

The interface component handles change detection and domain shift identification:

```python
class Interface:
    """Detects distribution changes and triggers adaptation"""
    
    def detect_change(self, data) -> bool:
        """Detect if adaptation is needed"""
        
    def get_adaptation_signal(self, data) -> float:
        """Get strength of adaptation signal"""
```

**Responsibilities:**
- Change point detection (CUSUM, statistical tests)
- Distribution shift measurement
- Triggering adaptation events

### 3. Adapter (`adapter.py`)

The adapter component handles local model adaptation using R-based GARCH models:

```python
class Adapter:
    """Handles local adaptation using R/rugarch models"""
    
    def fit(self, data):
        """Fit local adaptation model"""
        
    def update(self, new_data):
        """Update model with new data"""
        
    def forecast(self, horizon):
        """Generate local forecasts"""
```

**Responsibilities:**
- Interface with R/rugarch for volatility modeling
- Local parameter adaptation
- Short-term forecasting

### 4. Orchestrator (`orchestrator.py`)

The orchestrator ties everything together and manages the TTSTL workflow:

```python
class TTSTLOrchestrator:
    """Main TTSTL controller"""
    
    def __init__(self, backbone, interface, adapter):
        self.backbone = backbone
        self.interface = interface
        self.adapter = adapter
    
    def global_retrain(self, data):
        """Perform global backbone retraining"""
        
    def local_adapt(self, data):
        """Perform local adaptation"""
        
    def predict(self, data):
        """Generate predictions using both scales"""
```

**Responsibilities:**
- Coordinating global and local updates
- Managing the two-scale prediction process
- Handling the overall TTSTL workflow

## Data Flow

```
Raw Data
    ↓
[Preprocessing]
    ↓
[Backbone Feature Extraction]
    ↓
[Interface Change Detection] → [Adaptation Signal]
    ↓                              ↓
[Global Predictions]         [Local Adaptation]
    ↓                              ↓
[Combined Predictions] ← [Local Predictions]
```

## Time Scale Separation

### Fast Scale (Local)
- High-frequency updates
- Adapter-based modeling
- Short prediction horizons
- Responds to recent market regime changes

### Slow Scale (Global)
- Periodic retraining
- Backbone neural network
- Longer prediction horizons
- Captures long-term patterns

## Configuration

The system uses configuration files to manage different aspects:

```yaml
backbone:
  model_type: "cnn_lstm"
  input_dim: 10
  hidden_dim: 64
  num_layers: 2
  
interface:
  detection_method: "cusum"
  threshold: 0.05
  window_size: 100
  
adapter:
  model_type: "rugarch"
  garch_spec: "sGARCH"
  distribution: "std"
```

## Extension Points

The modular design allows for easy extension:

1. **New Backbones**: Implement the `Backbone` interface
2. **New Detection Methods**: Extend the `Interface` class
3. **New Adaptation Models**: Implement new `Adapter` classes
4. **Custom Orchestration**: Override `TTSTLOrchestrator` methods
