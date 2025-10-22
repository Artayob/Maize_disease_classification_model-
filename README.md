# Maize Disease Classification Model

Deep learning model for identifying common maize diseases from field images using transfer learning and CNN architectures.

## Project Overview

This project uses a two-stage training approach to build a robust maize disease classifier:

1. **Stage 1 - Baseline Training**: Train on clean PlantVillage dataset to learn fundamental disease features
2. **Stage 2 - Fine-tuning**: Adapt the model to real-world South African field conditions using Maize_in_Field dataset

### Diseases Detected

- Gray Leaf Spot (Cercospora)
- Common Rust
- Northern Leaf Blight
- Healthy

## Features

- Transfer learning with ResNet50, VGG16, or MobileNetV2
- Optimized for laptop/local development (GPU or CPU)
- Data augmentation for improved generalization
- Comprehensive evaluation with confusion matrices and per-class metrics
- TensorFlow Lite conversion for mobile deployment
- Complete Jupyter notebook with step-by-step guidance

## System Requirements

### Minimum
- **RAM**: 8 GB
- **Storage**: 15 GB free space
- **GPU**: Not required (CPU works, but slower)
- **OS**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: 3.8 or higher

### Recommended
- **RAM**: 16 GB
- **Storage**: 20 GB free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM (RTX 3070 or better)
- **Python**: 3.9 or 3.10

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Maize_disease_classification_model.git
cd Maize_disease_classification_model
```

### 2. Create Virtual Environment

**Windows:**
```cmd
python -m venv maize_env
maize_env\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv maize_env
source maize_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Kaggle API

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → API → "Create New API Token"
3. This downloads `kaggle.json`
4. Place it in the correct location:
   - **Windows**: `C:\Users\YourUsername\.kaggle\kaggle.json`
   - **Mac/Linux**: `~/.kaggle/kaggle.json`

### 5. Download Datasets

See [data/README.md](data/README.md) for detailed dataset download instructions.

**Quick start:**
```bash
# PlantVillage dataset
kaggle datasets download -d abdallahalidev/plantvillage-dataset

# Maize_in_Field dataset
kaggle datasets download -d [dataset-name]
```

Extract and organize according to the structure in `data/README.md`.

## Project Structure
```
Maize_disease_classification_model/
├── data/                          # Datasets (not in repo - download separately)
│   ├── plantvillage/             # Clean PlantVillage images
│   └── maize_in_field/           # Real-world field images
├── models/                        # Saved trained models (.h5, .tflite)
├── results/                       # Evaluation plots and metrics
├── logs/                          # TensorBoard training logs
├── notebooks/                     # Jupyter notebooks
│   └── maize_disease_classification_complete_guide.ipynb
├── maize_env/                     # Virtual environment (not in repo)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Usage

### Option 1: Jupyter Notebook (Recommended for Beginners)

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `notebooks/maize_disease_classification_complete_guide.ipynb`

3. Run cells sequentially from top to bottom

4. Follow the inline documentation and instructions

### Option 2: Command Line (Advanced)

Run the training script directly (if provided):
```bash
python train.py --model resnet50 --epochs 50 --batch-size 16
```

## Training

### Expected Training Times

| Hardware | Stage 1 (PlantVillage) | Stage 2 (Fine-tuning) | Total |
|----------|------------------------|------------------------|-------|
| RTX 3070/4070 GPU | 45-60 min | 25-35 min | ~1.5 hours |
| CPU (i7/i9) | 3-5 hours | 2-3 hours | ~6 hours |
| CPU (i5) | 5-8 hours | 3-5 hours | ~10 hours |

### Tips for Laptop Training

- **Reduce batch size** if you get memory errors (16→8→4)
- **Use MobileNetV2** for faster training on CPU
- **Train overnight** for long sessions
- **Keep laptop plugged in** and disable sleep mode
- **Close other applications** to free up RAM
- **Monitor temperature** - ensure good ventilation

## Evaluation Metrics

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-score**: Per-class and macro/micro averages
- **Confusion Matrix**: Shows which diseases are confused
- **Error Analysis**: Visual inspection of misclassified images

### Target Performance

- **PlantVillage test set**: 85-95% accuracy (clean data)
- **Maize_in_Field test set**: 75-85% accuracy (real-world data)

## Deployment

The final model can be converted to TensorFlow Lite for mobile deployment:
```python
# Conversion done in the notebook
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

The `.tflite` file can be integrated into Android/iOS apps for on-device inference.

## Results

Results are saved in the `results/` folder:

- Confusion matrices (PNG images)
- Training history plots (accuracy, loss curves)
- Classification reports (CSV/JSON)
- Model performance metrics

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size in the notebook configuration
BATCH_SIZE_STAGE1 = 8  # or 4
BATCH_SIZE_STAGE2 = 4  # or 2
```

### Slow Training on CPU

- Use MobileNetV2 instead of ResNet50
- Reduce number of epochs for testing (50→10)
- Use a smaller subset of data for initial experiments

### GPU Not Detected

1. Install CUDA Toolkit 12.3: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Verify installation: `nvcc --version`
3. Test TensorFlow GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### Kaggle API Issues

- Verify `kaggle.json` is in the correct location
- Check your Kaggle account is verified (phone number required)
- Test connection: `kaggle datasets list`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is for educational purposes. Please respect the licenses of the datasets used:

- PlantVillage: [Creative Commons licenses](https://www.plantvillage.org/)
- Maize_in_Field: Check Kaggle dataset page for license

## Acknowledgments

- PlantVillage dataset for clean baseline images
- Maize_in_Field dataset for real-world validation
- TensorFlow/Keras teams for the deep learning framework
- South African agricultural research community

## Contact

For questions or feedback, please open an issue on GitHub.

## Citation

If you use this project in your research, please cite:
```
@misc{maize_disease_classification,
  author = {Your Name},
  title = {Maize Disease Classification Model},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/Maize_disease_classification_model}
}
```