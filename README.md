# Deep Focused Sliding Windows for Fetus Brain Localization

## Overview

This repository contains the implementation of a hybrid model for fetal brain extraction from MRI scans using a cascade of UNets. The algorithm addresses challenges in fetal brain segmentation, particularly when training data is limited or lacks detailed annotations. By leveraging both high and low-resolution sliding windows and a cascade segmentation approach, this method improves brain detection and reduces false positives in low-contrast MRI scans.

## Algorithm Description

The algorithm combines two main strategies:
1. **Sliding Window Scan:** We use high and low-resolution sliding windows across the entire 3D MRI volume to identify potential brain regions. Large windows cover broader areas, while small windows focus on finer details.
2. **Cascade Segmentation:** Once the brain region is localized, we refine the segmentation using a cascade of smaller UNets (Models B and C) that operate at finer resolutions. This step helps reduce false positives and improves segmentation accuracy by focusing on challenging regions.

### Model Components:
- **Model A & D:** Large and small window scans for coarse segmentation and brain localization.
- **Model B & C:** Cascade refinement of localized regions using smaller UNets to achieve more accurate segmentation.

### Steps:
1. **Preprocessing:**
   - MRI images are first downsampled to match the fetal brain size range from training data.
   - Images are reshaped and padded/cropped to dimensions divisible by 2 to ensure compatibility with the network.
   - Label maps are generated using synthetic images augmented with brain tissue labels.
   
2. **Sliding Window Process:**
   - A sliding window approach is applied, using different step sizes (e.g., 64 for Models A and B, 32 for Models C and D) to scan the volume.
   - Voxels below a prediction threshold (e.g., 0.2) are discarded.
   - The largest connected component from this scan forms a bounding box around the brain region.

3. **Cascade Segmentation:**
   - The bounding box from the sliding window process is refined using a cascade of UNets.
   - Each model progressively improves segmentation accuracy, operating at finer scales.

### Datasets:
- **MGH Clinical Dataset**: HASTE MRI images of second and third-trimester fetuses.
- **FeTA Dataset**: Label maps of fetal brain structures used for training.
- **EPI Dataset**: Full-uterus 2D-EPI scans with mixed fetal positioning and resolutions.

## Installation

To set up the environment and install the required dependencies:

```bash
git clone https://github.com/yourusername/fetal-brain-segmentation.git
cd fetal-brain-segmentation
pip install -r requirements.txt
```

## Usage
Preprocess the data: Run the preprocessing script to reshape and downsample images.

```bash
python preprocess.py --input /path/to/data --output /path/to/preprocessed
```

## Train the model

Start training the cascade model using the preprocessed data.

```bash
python train.py --data /path/to/preprocessed --output /path/to/models
```

## Test the model


Evaluate the model on test images.
```bash
python test.py --model /path/to/models --data /path/to/testdata
```

## Results
Our method demonstrates superior segmentation performance, particularly for younger fetuses, when compared to baseline methods such as:

Ebner et al. (2020)
Keraudren et al. (2014)
Salehi et al. (2018)
We achieve higher Dice scores on clinical datasets, demonstrating the model's robustness across varying resolutions and fetal positions.

## Citation
If you use this code or model in your research, please cite:

```
@article{dadashkarimi2024,
  title={Search Wide, Focus Deep: Automated Fetal Brain Extraction With Sparse Training Data},
  author={Javid Dadashkarimi, Valeria Pena Trujillo, Camilo Jaimes, Lilla Z¨ollei, Malte Hoffmann},
  journal={arXiv},
  year={2024},
  publisher={arXiv}
}
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

