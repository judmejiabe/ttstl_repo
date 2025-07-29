# API Reference

## Core Classes

### TTSTLOrchestrator

Main controller class for the Two Time Scale Transfer Learning framework.

```python
class TTSTLOrchestrator:
    def __init__(self, backbone, interface, adapter, config=None)
```

**Parameters:**
- `backbone` (Backbone): Neural network backbone for feature extraction
- `interface` (Interface): Change detection interface
- `adapter` (Adapter): Local adaptation component
- `config` (dict, optional): Configuration dictionary

**Methods:**

#### global_retrain(data, targets=None, **kwargs)

Perform global retraining of the backbone model.

**Parameters:**
- `data` (array-like): Training features
- `targets` (array-like, optional): Training targets
- `**kwargs`: Additional training parameters

**Returns:**
- `dict`: Training metrics and history

#### local_adapt(data, targets=None, **kwargs)

Perform local adaptation using the adapter component.

**Parameters:**
- `data` (array-like): Recent data for adaptation
- `targets` (array-like, optional): Recent targets
- `**kwargs`: Additional adaptation parameters

**Returns:**
- `dict`: Adaptation results and metrics

#### predict(data, horizon=1, **kwargs)

Generate predictions using the two-scale approach.

**Parameters:**
- `data` (array-like): Input data for prediction
- `horizon` (int): Prediction horizon
- `**kwargs`: Additional prediction parameters

**Returns:**
- `array-like`: Predictions

#### save_model(path)

Save the complete TTSTL model.

**Parameters:**
- `path` (str): Path to save the model

#### load_model(path)

Load a saved TTSTL model.

**Parameters:**
- `path` (str): Path to the saved model

**Returns:**
- `TTSTLOrchestrator`: Loaded model instance

---

## Backbone Classes

### Backbone (Abstract Base Class)

Base class for all backbone models.

```python
class Backbone:
    def __init__(self)
```

**Abstract Methods:**

#### fit(data, targets, **kwargs)

Train the backbone model.

#### predict(data, **kwargs)

Generate predictions using the backbone.

#### save(path)

Save the backbone model.

#### load(path)

Load a saved backbone model.

### CNNLSTM

CNN-LSTM backbone for time series feature extraction.

```python
class CNNLSTM(Backbone):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=2,
        cnn_filters=32,
        kernel_size=3,
        dropout=0.1,
        **kwargs
    )
```

**Parameters:**
- `input_dim` (int): Number of input features
- `hidden_dim` (int): Hidden dimension size
- `num_layers` (int): Number of LSTM layers
- `cnn_filters` (int): Number of CNN filters
- `kernel_size` (int): CNN kernel size
- `dropout` (float): Dropout rate

**Methods:**

#### forward(x)

Forward pass through the CNN-LSTM model.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch, seq_len, features)

**Returns:**
- `torch.Tensor`: Output predictions

### CNNLSTMDataset

PyTorch dataset wrapper for CNN-LSTM training.

```python
class CNNLSTMDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, sequence_length=50)
```

**Parameters:**
- `features` (array-like): Input features
- `targets` (array-like): Target values
- `sequence_length` (int): Length of input sequences

---

## Interface Classes

### Interface

Change detection and adaptation triggering interface.

```python
class Interface:
    def __init__(
        self,
        detection_method='cusum',
        threshold=0.05,
        window_size=100,
        **kwargs
    )
```

**Parameters:**
- `detection_method` (str): Change detection method ('cusum', 'ks_test', 'statistical')
- `threshold` (float): Detection threshold
- `window_size` (int): Window size for detection

**Methods:**

#### detect_change(data)

Detect if a distribution change has occurred.

**Parameters:**
- `data` (array-like): Recent data observations

**Returns:**
- `bool`: True if change detected, False otherwise

#### get_adaptation_signal(data)

Get the strength of the adaptation signal.

**Parameters:**
- `data` (array-like): Recent data observations

**Returns:**
- `float`: Adaptation signal strength (0-1)

#### update_reference(data)

Update the reference distribution for change detection.

**Parameters:**
- `data` (array-like): New reference data

---

## Adapter Classes

### Adapter

Local adaptation component using R-based models.

```python
class Adapter:
    def __init__(
        self,
        model_type='rugarch',
        garch_spec='sGARCH',
        distribution='std',
        **kwargs
    )
```

**Parameters:**
- `model_type` (str): Type of adaptation model
- `garch_spec` (str): GARCH specification for rugarch
- `distribution` (str): Error distribution

**Methods:**

#### fit(data, **kwargs)

Fit the local adaptation model.

**Parameters:**
- `data` (array-like): Training data
- `**kwargs`: Additional fitting parameters

#### update(new_data, **kwargs)

Update the model with new observations.

**Parameters:**
- `new_data` (array-like): New data for updating
- `**kwargs`: Additional update parameters

#### forecast(horizon=1, **kwargs)

Generate local forecasts.

**Parameters:**
- `horizon` (int): Forecast horizon
- `**kwargs`: Additional forecasting parameters

**Returns:**
- `array-like`: Forecast values

#### get_parameters()

Get current model parameters.

**Returns:**
- `dict`: Current model parameters

---

## Utility Functions

### preprocess_mbo_data_cnn_lstm(path, delta, quantile=0.005)

Preprocess market microstructure data for CNN-LSTM training.

**Parameters:**
- `path` (str): Path to data file
- `delta` (float): Sampling time interval
- `quantile` (float): Winsorization quantile

**Returns:**
- `tuple`: (features, targets) as pandas DataFrames

### download_from_bucket(blob_name, destination_file_name, bucket_name)

Download data from Google Cloud Storage.

**Parameters:**
- `blob_name` (str): Name of the blob to download
- `destination_file_name` (str): Local destination filename
- `bucket_name` (str): GCS bucket name

### import_to_pandas(blob_name, destination_file_name, bucket_name)

Download and import data directly to pandas DataFrame.

**Parameters:**
- `blob_name` (str): Name of the blob to download
- `destination_file_name` (str): Temporary filename
- `bucket_name` (str): GCS bucket name

**Returns:**
- `pd.DataFrame`: Imported data

### get_datasets_cnn_lstm(features, targets, sequence_length=50, train_split=0.8)

Create train/validation datasets for CNN-LSTM training.

**Parameters:**
- `features` (array-like): Input features
- `targets` (array-like): Target values
- `sequence_length` (int): Length of input sequences
- `train_split` (float): Fraction of data for training

**Returns:**
- `tuple`: (train_loader, val_loader) as PyTorch DataLoaders

---

## Configuration Schema

### Backbone Configuration

```yaml
backbone:
  model_type: str              # Model type ("cnn_lstm")
  input_dim: int              # Number of input features
  hidden_dim: int             # Hidden dimension size
  num_layers: int             # Number of LSTM layers
  cnn_filters: int            # Number of CNN filters
  kernel_size: int            # CNN kernel size
  dropout: float              # Dropout rate
  learning_rate: float        # Learning rate
  batch_size: int             # Batch size
  epochs: int                 # Number of training epochs
```

### Interface Configuration

```yaml
interface:
  detection_method: str       # Detection method
  threshold: float            # Detection threshold
  window_size: int           # Window size
  warmup_period: int         # Warmup period
  significance_level: float   # For statistical tests
```

### Adapter Configuration

```yaml
adapter:
  model_type: str            # Adapter model type
  garch_spec: str           # GARCH specification
  distribution: str          # Error distribution
  mean_model: str           # Mean model specification
  update_frequency: int      # Update frequency
  window_size: int          # Rolling window size
```
