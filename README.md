
---

## AI vs Real Classification

### Location:
`AI vs Real/Classifier.ipynb`

### Overview:
This notebook contains:
- **Data Loading**: Fetch and preprocess data from various sources.
- **Dependencies**: Includes libraries like `keras` and `scikit-learn`.
- **Model Training**: Implements a machine learning classifier to differentiate between AI-generated and real data.

### How to Run:
1. Install dependencies: `pip install -r requirements_ai_vs_real.txt`.
2. Open the notebook: `AI vs Real/Classifier.ipynb`.
3. Follow the instructions to load data, train the model, and evaluate performance.

---

## GAN Training

### Location:
`GAN/Training.ipynb`

### Overview:
This notebook focuses on training a Generative Adversarial Network (GAN):
- **Generator**: Produces synthetic data based on random noise.
- **Discriminator**: Evaluates whether the input is real or synthetic.
- **Loss Functions**: Implements Wasserstein Loss and Gradient Penalty for stable training.
- **Training Loop**: Iteratively trains the GAN to improve synthetic data quality.

### How to Run:
1. Install dependencies: `pip install -r requirements_custom_gan.txt`.
2. Open the notebook: `GAN/Training.ipynb`.
3. Execute the cells to train the GAN model on your dataset.

---

## Requirements

### For AI vs Real Classification
- `numpy`
- `pandas`
- `tqdm`
- `scikit-learn`
- `keras`
- `keras-preprocessing`
- `tensorflow`

### For GAN Training
- `torch`
- `torchvision`
- `matplotlib`
- `Pillow`
