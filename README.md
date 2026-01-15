# Signature Verification System

A comprehensive signature verification system using Siamese neural networks for authenticating handwritten signatures and detecting forgeries.

## Overview

This project implements a signature verification system that uses deep learning techniques to compare signature images and determine their authenticity. The system combines a Siamese neural network architecture with traditional image processing techniques to provide robust signature verification capabilities.

## Features

- Siamese neural network for signature comparison
- Forgery detection using multiple heuristics
- Signature drift detection over time
- Risk scoring for verification decisions
- Web-based interface for easy use
- Batch processing capabilities

## Architecture

The system is organized into the following modules:

### Dataset Structure
```
dataset/
├── person1/
│   ├── genuine/
│   │   ├── g1.png
│   │   ├── g2.png
│   │   └── g3.png
│   └── forged/
│       ├── f1.png
│       ├── f2.png
│       └── f3.png
...
```

### Core Modules

- **models**: Contains the Siamese network architecture and trained model
- **preprocessing**: Image preprocessing and feature extraction
- **training**: Training pipeline and data generators
- **verification**: Signature comparison and risk analysis
- **app**: Flask web application
- **utils**: Configuration and helper functions

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

```bash
cd app
python app.py
```

The application will be available at `http://localhost:5000`

### Training the Model

```bash
cd training
python train_model.py --dataset_path ../dataset --epochs 50
```

### Command Line Verification

```bash
python main.py --genuine_path path/to/genuine/signature.png --test_path path/to/test/signature.png
```

## Components

### Siamese Network
The core of the system uses a Siamese network architecture based on a pre-trained VGG16 model with custom top layers for signature-specific features.

### Preprocessing Pipeline
- Image normalization and resizing
- Noise reduction
- Contrast enhancement
- Background removal

### Verification Process
1. **Similarity Check**: Uses the trained Siamese network to compare signatures
2. **Forgery Detection**: Analyzes the signature for potential forgery indicators
3. **Drift Analysis**: Compares against historical signatures to detect changes
4. **Risk Scoring**: Combines all factors into a final risk score

## Risk Scoring

The system calculates a risk score based on multiple factors:
- Signature similarity (40%)
- Forgery indicators (30%)
- Signature drift (20%)
- Image quality (10%)

Risk levels:
- **LOW** (< 30%): Accept signature
- **MEDIUM** (30-60%): Review signature
- **HIGH** (> 60%): Reject signature

## File Structure

```
signature-verification/
├── dataset/                 # Signature images organized by person and type
├── models/                  # Neural network architecture and weights
├── preprocessing/           # Image processing and feature extraction
├── training/                # Training scripts and data generators
├── verification/            # Signature comparison and analysis
├── app/                     # Web application
├── utils/                   # Configuration and helper functions
├── results/                 # Output and logs
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── main.py                 # Command-line entry point
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.