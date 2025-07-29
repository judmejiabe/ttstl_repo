# TTSTL: Two Time Scale Transfer Learning

A Python package for implementing Two Time Scale Transfer Learning (TTSTL) for time series forecasting, particularly focused on financial market data.

## Overview

TTSTL is a novel approach that combines:
- **Two Time Scale Decomposition**: Separating time series into fast and slow components
- **Transfer Learning**: Leveraging pre-trained models for new domains/markets
- **Adaptive Learning**: Continuous model updates with new data

## Features

- 🔄 **Two-Scale Architecture**: Decompose time series into multiple temporal scales
- 🧠 **Neural Backbone**: CNN-LSTM based backbone for feature extraction
- 🔌 **Modular Interface**: Clean separation between components
- 📈 **Financial Focus**: Optimized for limit order book and market microstructure data
- 🚀 **Easy Integration**: Simple API for training, adaptation, and forecasting

## Installation

### From Source (Development)

```bash
git clone https://github.com/judmejiabe/ttstl_repo.git
cd ttstl_repo
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev,docs,notebooks]"
```

## Quick Start

```python
import ttstl
from ttstl import TTSTLModel, Backbone, Interface, Adapter

# Initialize the TTSTL model
model = TTSTLModel(
    backbone_config={'input_dim': 10, 'hidden_dim': 64},
    interface_config={'detection_method': 'cusum'},
    adapter_config={'model_type': 'rugarch'}
)

# Train on source domain
model.fit(source_data, source_targets)

# Adapt to target domain
model.adapt(target_data, target_targets)

# Make predictions
predictions = model.predict(new_data)
```

## Project Structure

```
ttstl_repo/
├── src/ttstl/           # Core package
├── scripts/             # CLI tools
├── notebooks/           # Examples and tutorials
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{mejia2025ttstl,
  title={Two Time Scale Transfer Learning for Financial Time Series Forecasting},
  author={Mejia Becerra, Juan Diego},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

- Author: Juan Diego Mejia Becerra
- Email: juanmejia.7565@gmail.com
