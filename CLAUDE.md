# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SJ_Haar_CNV is a Haar wavelet-based CNV (Copy Number Variation) detection tool developed during the 2024 St. Jude Children's Research Hospital BioHackathon. The project implements a three-stage pipeline: base search, decomposition, and reporting.

## Environment Setup

This project uses conda for dependency management:

```bash
# Create and activate conda environment
conda env create -f haar_env.yml
conda activate haar_env
```

The environment includes Python 3.11 with key dependencies:
- NumPy 2.0.0 for numerical operations
- SciPy 1.14.0 for optimization and signal processing
- Pandas 2.2.2 for data manipulation
- Plotly 5.22.0 for interactive visualizations
- scikit-learn, statsmodels, numba for ML and statistical analysis

## Architecture

### Core Components

The codebase is organized into three main modules in `SJ_Haar_CNV/`:

1. **base_search.py** - Haar wavelet basis generation
   - Implements Haar wavelet basis vector construction
   - `generate_haar_basis()`: Main entry point that recursively finds optimal breakpoints in signals
   - `choose_break()`: Selects breakpoints using a configurable p0 parameter (range [0.5, 1)) to control search window
   - Wavelet representation: `[depth, unused, start, (high_value, high_length), (low_value, low_length)]`

2. **decomposition.py** - Signal decomposition and thresholding
   - `decompose()`: Decomposes signals into wavelet coefficients with automatic or manual thresholding
   - `estimate_threshold()`: Uses MAD (Median Absolute Deviation) with k=1.4826 for robust threshold estimation
   - `haar_basis()`: Converts compact wavelet representation into full basis matrix
   - `generate_function_from_wavelets()`: Reconstructs signals from coefficients
   - `rle()`: Run-length encoding for compact representation of step functions

3. **report.py** - Visualization
   - `visualize_data()`: Creates interactive Plotly figures with raw signal, transformed signal, and smoothed noise bands
   - Uses Savitzky-Golay filtering (default: window=100, polyorder=3) for noise estimation
   - `add_noise()`: Adds confidence bands around the transformed signal

4. **utils.py** - Helper functions
   - `parameterize_wavelet()`: Converts length-based representation to start/end coordinates
   - `expand_wavelet()`: Expands RLE representation to full arrays
   - `visualize_wavelet()`: Quick visualization of wavelet data

### Pipeline Flow

1. **Base Search**: `generate_haar_basis()` recursively partitions the signal into Haar wavelets up to a specified depth or minimum segment length
2. **Decomposition**: `decompose()` projects the signal onto the wavelet basis and applies soft thresholding to denoise
3. **Report**: `visualize_data()` creates interactive plots showing the original signal, denoised result, and uncertainty estimates

## Development Workflow

### Running Tests

The project uses pytest (though not installed in the base environment):

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_decomposition.py

# Run with verbose output
pytest -v tests/
```

Test files are located in `tests/`:
- `test_base_search.py` - Tests for basis generation
- `test_decomposition.py` - Tests for signal decomposition
- `test_report.py` - Tests for visualization functions
- `test_data.py` - Test data generation utilities

### Interactive Development

Primary development happens in Jupyter notebooks:
- `Base_Playground.ipynb` - Experimentation with basis search algorithms
- `demo.ipynb` - Full pipeline demonstrations

To run notebooks:
```bash
jupyter notebook
```

## Key Algorithm Parameters

### generate_haar_basis()
- `p0`: Controls breakpoint search window (default 0.95). Higher values restrict search to more central locations
- `length`: Minimum segment length before recursion stops (default 20)
- `depth_limit`: Maximum recursion depth (optional, None = unlimited)

### decompose()
- `threshold`: Coefficient threshold for denoising (None = auto-estimate)
- `k`: MAD scaling factor for threshold estimation (default 1.4826 for Gaussian noise)

### visualize_data()
- `noise_settings`: Dictionary with 'window' and 'polyorder' for Savitzky-Golay filtering
- `markers`: Plotly marker styling for raw data points
- `line`: Plotly line styling for transformed signal

## Code Conventions

- Wavelet tuples use format: `[depth, unused, start, (high_val, high_len), (low_val, low_len)]`
- Signal indexing follows Python conventions (0-based, exclusive end)
- Breakpoints `b` satisfy `s < b < e` for segment `[s, e)`
- The first coefficient (index 0) represents the mean and is always preserved in decomposition
