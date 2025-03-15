# Low-Light Image Enhancement (LLIE) with U-Net

This repository contains PyTorch code for low-light image enhancement using a U-Net-based architecture augmented with Squeeze-and-Excitation (SE) and spatial attention blocks.

## Overview

- **Goal**: Enhance underexposed or low-light images by improving brightness, contrast, and preserving details.  
- **Approach**: A modified U-Net with attention mechanisms (SE and spatial) is trained on paired low-light and normal-light images to learn the enhancement mapping.  
- **Dataset**: Assumes the LOL (or similar) dataset in a folder structure as described below.  
- **Pretrained Model**: A pretrained model checkpoint is available for download (see link below).

## Features

1. **Attention Blocks**: SEBlock (channel attention) and SpatialAttn (spatial attention) to focus on important features.  
2. **Combined L1 + SSIM Loss**: Encourages pixel-wise accuracy and structural similarity.  
3. **Residual Approach (Optional)**: You can enable a residual output (network predicts a delta) to avoid over-smoothing.  
4. **Perceptual and Edge Loss (Optional)**: The code can be extended with advanced losses for sharper details.

## Dataset Structure

The LOL dataset can be found at https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

## Model

The best model that we trained and saved can be found at: https://ucla.box.com/s/5g03rl3ves3t6d0iajpxrbl9uw9tkunt. The file exceeds the 100 MB limit per file limit on Github so we were forced to upload this file to Box. If you would like to test on this model, simply download the .pth file into this respository's folder on your local machine and run the model in test mode.
