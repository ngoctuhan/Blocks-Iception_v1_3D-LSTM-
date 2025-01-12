# I3D-LSTM Network for Video Action Recognition

This repository contains an implementation of the I3D-LSTM architecture for video action recognition using TensorFlow/Keras. The model combines the Inception-3D (I3D) architecture pre-trained on Kinetics dataset with LSTM layers for temporal modeling of video sequences.

## Architecture Overview

The model consists of two main components:
- I3D (Inception-3D) network pre-trained on Kinetics dataset for spatial-temporal feature extraction
- LSTM layers for modeling temporal dependencies across video segments

## Requirements

```
tensorflow==2.1
opencv-python==4.1
numpy
matplotlib
```

## Project Structure

```
├── models/
│   ├── networks.py        # Contains I3D and I3D-LSTM model implementations
├── utils/
│   ├── params.py         # Parameter configurations
│   ├── pre_process.py    # Data preprocessing utilities
│   ├── videoto3D.py      # Video frame extraction
│   └── sequence.py       # Data sequence generator for training
├── training.py           # Main training script
```

## Model Details

The implementation includes:
- Pre-trained I3D model (both RGB and optical flow variants)
- Support for both Kinetics and ImageNet + Kinetics pre-trained weights
- Customizable input shapes and model parameters
- TimeDistributed wrapper for processing video segments
- Double LSTM layers (512 and 256 units) for temporal modeling

## Training

The training process uses:
- SGD optimizer with momentum
- Learning rate scheduling (3 phases)
- Categorical crossentropy loss
- Data augmentation through sequence generator
- Batch-wise training for memory efficiency

### Learning Rate Schedule:
- First 10 epochs: Initial learning rate
- Epochs 11-15: Reduced learning rate
- Epochs 16+: Final reduced learning rate

## Usage

1. Set up your data path and parameters in `utils/params.py`

2. Run the training script:
```bash
python training.py
```

The model will save checkpoints when validation accuracy improves.

## Pre-trained Models

The implementation supports loading pre-trained I3D weights:
- RGB model weights: Pre-trained on ImageNet + Kinetics
- Optical Flow model weights: Pre-trained on ImageNet + Kinetics

Weights are automatically downloaded from the official repository.

## Dataset Preparation

The code expects video data to be organized in the following structure:
```
data_path/
    ├── class1/
    │   ├── video1.avi
    │   ├── video2.avi
    │   └── ...
    ├── class2/
    │   ├── video1.avi
    │   ├── video2.avi
    │   └── ...
    └── ...
```

## Model Configuration

Key parameters that can be configured in `params.py`:
- Number of frames per segment
- Input image size
- Batch size
- Learning rates for different phases
- Number of output classes
- Number of LSTM blocks

## Citation

If you use this implementation in your research, please cite:

```
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition? a new model and the kinetics dataset},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6299--6308},
  year={2017}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
