# Backbone Import and Reuse Guide

## Overview

The TTSTL framework now supports importing and reusing pretrained backbone modules across different orchestrator instances. This feature enables significant computational savings and supports various deployment patterns for production and research use cases.

## Key Benefits

- ⚡ **Performance**: Skip expensive neural network training when reusing models
- 🔄 **Reusability**: Train once, deploy multiple times across different datasets
- 💾 **Efficiency**: Reduce computational costs and deployment time
- 🎯 **Consistency**: Maintain consistent feature extraction across experiments
- 🏗️ **Modularity**: Support flexible architecture patterns

## How It Works

### 1. Training Phase
First, create an orchestrator and train backbone modules normally:

```python
from ttstl.orchestrator import Orchestrator, OrchestratorConfig

# Configure training orchestrator (normal configuration)
training_config = OrchestratorConfig(
    session_start=pd.Timestamp('2024-01-01 09:30:00'),
    session_finish=pd.Timestamp('2024-01-01 16:00:00'),
    # ... other configuration parameters
)

# Create and run training orchestrator
training_orchestrator = Orchestrator(runs_dir, training_config)
training_results = training_orchestrator.global_update(event_df, phys_df)
```

### 2. Import Phase
Create a new orchestrator and import the pretrained backbone modules:

```python
# Configure testing orchestrator with skip_backbone_training=True
testing_config = OrchestratorConfig(
    session_start=pd.Timestamp('2024-01-02 09:30:00'),
    session_finish=pd.Timestamp('2024-01-02 16:00:00'),
    skip_backbone_training=True,  # 🔑 Key flag!
    # ... other configuration parameters (should match training config)
)

# Create testing orchestrator
testing_orchestrator = Orchestrator(runs_dir, testing_config)

# Import pretrained backbone modules
testing_orchestrator.import_backbone_modules_from_orchestrator(training_orchestrator)

# Run global update (will skip backbone training)
testing_results = testing_orchestrator.global_update(new_event_df, new_phys_df)
```

## Configuration Requirements

### Essential Settings

1. **skip_backbone_training=True**: Must be set in the target orchestrator configuration
2. **Matching Architecture**: Both orchestrators should have compatible configurations for:
   - `bb_lookback` and `bb_forward` (backbone window sizes)
   - `target_col_name` (target variable name)
   - Neural network architecture parameters

### Recommended Settings

- Keep `update_freq`, `quantile`, and other processing parameters consistent
- Ensure both orchestrators use the same data preprocessing approach

## Use Cases

### 1. Production Deployment
```python
# One-time training on historical data
production_backbone = train_backbone_modules(historical_data)

# Real-time deployment (fast startup)
realtime_orchestrator = create_orchestrator(skip_backbone_training=True)
realtime_orchestrator.import_backbone_modules_from_orchestrator(production_backbone)
```

### 2. Research Workflows
```python
# Train on representative dataset
base_orchestrator = train_on_representative_data()

# Test across multiple time periods
for test_period in test_periods:
    test_orchestrator = create_test_orchestrator(test_period, skip_backbone_training=True)
    test_orchestrator.import_backbone_modules_from_orchestrator(base_orchestrator)
    results[test_period] = test_orchestrator.global_update(test_data[test_period])
```

### 3. A/B Testing
```python
# Fixed backbone, different adapter configurations
backbone_orchestrator = train_backbone_once()

for adapter_config in adapter_configs:
    test_orchestrator = create_orchestrator(adapter_config, skip_backbone_training=True)
    test_orchestrator.import_backbone_modules_from_orchestrator(backbone_orchestrator)
    results[adapter_config] = test_orchestrator.global_update(test_data)
```

## Method Reference

### `import_backbone_modules_from_orchestrator(source_orchestrator)`

Imports pretrained backbone modules from another orchestrator instance.

**Parameters:**
- `source_orchestrator` (Orchestrator): The orchestrator instance containing pretrained backbone modules

**Behavior:**
- Validates that the source orchestrator has trained backbone modules
- Copies backbone instances to the current orchestrator's stacks
- Maintains the mapping between backbone bars and modules
- Logs the import process for debugging

**Requirements:**
- Source orchestrator must have completed `global_update()` with trained backbones
- Target orchestrator must have `skip_backbone_training=True` in configuration
- Both orchestrators should have compatible architecture configurations

## Error Handling

Common issues and solutions:

### `RuntimeError: Source orchestrator has no trained backbone modules`
- Ensure the source orchestrator has completed `global_update()` successfully
- Check that backbone training was not skipped in the source orchestrator

### `RuntimeError: This orchestrator is not configured to skip backbone training`
- Set `skip_backbone_training=True` in the target orchestrator's configuration

### Architecture Mismatch Errors
- Ensure both orchestrators have the same `bb_lookback`, `bb_forward`, and neural network parameters
- Verify that the target variable name and data preprocessing are consistent

## Examples

See the following example scripts:

1. **`scripts/backbone_import_example.py`**: Standalone demonstration of backbone import functionality
2. **`scripts/synthetic_study.py`**: Integration with existing workflow (run with `--demo-backbone-import`)

Run the examples:
```bash
# Standalone example
python scripts/backbone_import_example.py

# Integrated example
python scripts/synthetic_study.py --demo-backbone-import
```

## Performance Considerations

### Memory Usage
- Imported backbone modules are loaded into memory
- For large numbers of backbone modules, consider memory constraints
- The import process copies model instances, not just references

### Computational Savings
- Typical speedup: 2-10x faster (depending on backbone complexity)
- Greatest benefit when backbone training is computationally expensive
- Interface and adapter steps still run normally

### Best Practices
1. Train backbone modules on representative, high-quality data
2. Validate imported models on known test cases before production use
3. Monitor performance to ensure imported models maintain accuracy
4. Keep configurations well-documented for reproducibility

## Technical Details

### Implementation Notes
- Backbone modules are copied using deep copy to ensure independence
- Stack mapping preserves the relationship between backbone bars and modules
- The import process validates source orchestrator state before copying
- Error handling provides clear feedback for common configuration issues

### Limitations
- Only backbone modules are imported (interface and adapter modules are retrained)
- Requires compatible orchestrator configurations
- Source and target orchestrators should use the same TTSTL version

---

For more information, see the TTSTL documentation or contact the development team.
