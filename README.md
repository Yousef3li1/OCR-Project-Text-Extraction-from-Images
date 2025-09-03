# OCR Project – Text Extraction from Images

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org)
[![Tesseract](https://img.shields.io/badge/Tesseract-5.0%2B-orange.svg)](https://tesseract-ocr.github.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Overview

An advanced **Optical Character Recognition (OCR)** system that extracts text from images and scanned documents with high accuracy, even from low-quality sources. This project supports both **English and Arabic** text recognition and combines traditional OCR with modern deep learning techniques.

## ✨ Features

- 🔍 **High-accuracy text extraction** from images and PDFs
- 🌐 **Multilingual support** (Arabic & English)
- 🖼️ **Advanced image preprocessing** with noise reduction
- 🧠 **Deep learning enhancement** for complex text recognition
- 📱 **Flexible deployment** options (CLI, Jupyter, Web App)
- ⚡ **Batch processing** capabilities
- 📊 **Real-time performance** monitoring

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Image Processing** | OpenCV | Preprocessing, noise reduction, enhancement |
| **OCR Engine** | Tesseract OCR | Core text recognition |
| **Deep Learning** | TensorFlow/Keras | Advanced text detection and recognition |
| **Data Processing** | NumPy, Pandas | Data manipulation and analysis |
| **Visualization** | Matplotlib | Results visualization and debugging |
| **Web Framework** | Flask/FastAPI/Streamlit | Optional web interface |

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Tesseract OCR 5.0+
- OpenCV 4.5+

### Python Dependencies
```bash
opencv-python>=4.5.0
pytesseract>=0.3.9
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
Pillow>=8.3.0
scikit-image>=0.18.0
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ocr-project.git
   cd ocr-project
   ```

2. **Install Tesseract OCR**
   
   **Windows:**
   ```bash
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   # Or using chocolatey:
   choco install tesseract
   ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt update
   sudo apt install tesseract-ocr tesseract-ocr-ara
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

#### Command Line Interface
```bash
# Extract text from single image
python ocr_extract.py --image path/to/image.jpg --lang eng+ara

# Batch processing
python ocr_extract.py --folder path/to/images/ --output results.txt

# With preprocessing options
python ocr_extract.py --image image.jpg --denoise --enhance-contrast --lang ara
```

#### Python Script
```python
from ocr_system import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor(languages=['eng', 'ara'])

# Extract text from image
result = ocr.extract_text('path/to/image.jpg')
print(f"Extracted text: {result['text']}")
print(f"Confidence: {result['confidence']}%")

# Extract with preprocessing
result = ocr.extract_text(
    'path/to/image.jpg',
    preprocess=True,
    denoise=True,
    enhance_contrast=True
)
```

#### Jupyter Notebook
```python
import cv2
import matplotlib.pyplot as plt
from ocr_system import OCRProcessor, ImagePreprocessor

# Load and display image
image = cv2.imread('sample.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# Preprocess image
preprocessor = ImagePreprocessor()
cleaned_image = preprocessor.enhance_image(image)

# Extract text
ocr = OCRProcessor()
text = ocr.extract_text_from_array(cleaned_image)
print(text)
```

## 📁 Project Structure

```
ocr-project/
├── src/
│   ├── ocr_system.py          # Main OCR processing class
│   ├── image_preprocessor.py  # Image enhancement utilities
│   ├── deep_learning/
│   │   ├── text_detector.py   # Deep learning text detection
│   │   └── models/            # Trained model files
│   └── utils/
│       ├── language_utils.py  # Language detection helpers
│       └── file_utils.py      # File I/O utilities
├── notebooks/
│   ├── demo.ipynb            # Interactive demo
│   └── training.ipynb        # Model training notebook
├── tests/
│   ├── test_ocr.py           # Unit tests
│   └── sample_images/        # Test images
├── web_app/
│   ├── app.py                # Flask/Streamlit web interface
│   └── templates/            # HTML templates
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 Configuration

### Language Configuration
```python
# Supported languages
SUPPORTED_LANGUAGES = {
    'eng': 'English',
    'ara': 'Arabic',
    'fra': 'French',
    'deu': 'German'
}

# OCR Configuration
OCR_CONFIG = {
    'page_segmentation_mode': 6,  # PSM_SINGLE_UNIFORM_BLOCK
    'ocr_engine_mode': 3,         # OEM_DEFAULT
    'confidence_threshold': 60
}
```

### Preprocessing Options
```python
PREPROCESSING_CONFIG = {
    'denoise': True,
    'enhance_contrast': True,
    'deskew': True,
    'resize_factor': 2.0,
    'gaussian_blur_kernel': (1, 1)
}
```

## 📊 Performance

### Benchmarks
| Image Type | Accuracy | Processing Time |
|------------|----------|----------------|
| High-quality scan | 98.5% | 0.8s |
| Low-quality photo | 87.2% | 1.2s |
| Handwritten text | 79.3% | 1.5s |
| Arabic text | 92.1% | 1.1s |

### Optimization Tips
- Use `--enhance-contrast` for low-light images
- Enable `--denoise` for noisy scanned documents
- Increase image resolution for small text
- Use appropriate language models for better accuracy

## 🌐 Web Interface

### Streamlit App
```bash
streamlit run web_app/streamlit_app.py
```

### Flask API
```bash
python web_app/flask_app.py
```

**API Endpoints:**
- `POST /api/extract` - Extract text from uploaded image
- `GET /api/languages` - Get supported languages
- `GET /api/health` - Health check

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_ocr.py::test_english_extraction

# Generate coverage report
python -m pytest --cov=src tests/
```

## 🚀 Deployment

### Docker
```dockerfile
FROM python:3.9-slim

# Install Tesseract
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-ara

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Run application
CMD ["python", "app.py"]
```

### Cloud Deployment
- **AWS:** Use Lambda + API Gateway for serverless deployment
- **Google Cloud:** Deploy on Cloud Run with container
- **Azure:** Use Container Instances or App Service

## 🔮 Future Enhancements

- [ ] **Additional Languages:** Support for Chinese, Japanese, Russian
- [ ] **Advanced Models:** Integration with CRNN and Transformer models
- [ ] **Real-time OCR:** Camera feed processing
- [ ] **Mobile App:** React Native or Flutter implementation
- [ ] **API Rate Limiting:** Production-ready API with authentication
- [ ] **Database Integration:** Store and manage extracted text
- [ ] **Advanced Analytics:** Text analysis and insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/
```


## 🙏 Acknowledgments

- **Tesseract OCR Team** for the excellent OCR engine
- **OpenCV Community** for computer vision tools
- **TensorFlow Team** for deep learning framework
- **Contributors** who helped improve this project


---

⭐ **Star this repository if it helped you!**
