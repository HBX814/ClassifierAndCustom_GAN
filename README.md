
---

## AI vs Real Classification

### Location:
`AI vs Real/Classifier.ipynb`

# Vision Transformer (ViT) Image Classification

This project leverages the Vision Transformer (ViT) model for classifying images into different categories. The code includes data preprocessing, model training, evaluation, and testing using the Hugging Face `transformers` library. The model is fine-tuned using a dataset stored locally, and predictions are made on a separate test dataset.

## Requirements

Before running the code, ensure that the following dependencies are installed:

1. **TensorFlow**: For model training and Keras functionality.
2. **tqdm**: To visualize the progress during training and processing.
3. **scikit-learn**: For Label Encoding and dataset handling.
4. **transformers**: To use pre-trained Vision Transformer (ViT) models from Hugging Face.
5. **datasets**: For loading and processing datasets.
6. **accelerate**: To optimize model training and inference performance.
7. **PIL (Pillow)**: For image manipulation and resizing.
8. **torch and torchvision**: For tensor operations and model predictions.

### Installation

To install the required dependencies, use the following commands:

```bash
pip install tensorflow tqdm scikit-learn transformers datasets accelerate torch torchvision
