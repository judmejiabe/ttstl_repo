from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ttstl",
    version="0.1.0",
    author="Juan Diego Mejia Becerra",
    author_email="juanmejia.7565@gmail.com",
    description="Two Time Scale Transfer Learning package for time series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/judmejiabe/ttstl_repo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "google-cloud-storage>=2.0.0",
        "rpy2>=3.4.0",  # For R integration (rugarch)
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "plotly>=5.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ttstl-global-retrain=scripts.run_global_retrain:main",
            "ttstl-local-adapt=scripts.run_local_adapt:main",
            "ttstl-forecast=scripts.forecast:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
