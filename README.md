# AquaVisionAI: Marine Ecosystem Monitoring System

AquaVisionAI is an advanced system for automated analysis of underwater imagery for marine ecosystem monitoring using AI. The system integrates three specialized models to provide comprehensive analysis of marine environments.

![AquaVisionAI Logo](https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/logo.png)

## Features

- **Marine Species Identification**: Identify 23 different marine species with confidence scores
- **Coral Health Assessment**: Evaluate coral health (healthy vs. bleached) with detailed analysis
- **Pollution Detection**: Detect and classify various types of underwater pollution
- **Image Enhancement**: Optimized preprocessing for underwater imagery
- **Interactive Dashboard**: User-friendly interface for uploading and analyzing images
- **Analysis History**: Track and visualize results over time
- **Data Export**: Export analysis results for further research

## System Architecture

AquaVisionAI consists of three main components:

1. **Image Preprocessing Module**: Enhances underwater images for better analysis
2. **AI Models**:
   - Marine Species Identification Model (MobileNetV2-based)
   - Coral Health Assessment Model (MobileNetV2-based)
   - Pollution Detection Model (YOLOv8-based)
3. **Web Interface**: Streamlit-based dashboard for user interaction

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/AquaVisionAI.git
   cd AquaVisionAI
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Open the web interface in your browser
2. Upload one or more underwater images
3. View the analysis results, including:
   - Identified marine species with confidence scores
   - Coral health assessment
   - Detected pollution with bounding boxes
4. Export analysis history for further research

## Model Details

### Marine Species Identification Model

- Architecture: MobileNetV2
- Classes: 23 marine species
- Input size: 224x224 pixels
- Output: Species classification with confidence scores

### Coral Health Assessment Model

- Architecture: MobileNetV2
- Classes: Healthy vs. Bleached
- Input size: 224x224 pixels
- Output: Health status with confidence score

### Pollution Detection Model

- Architecture: YOLOv8
- Classes: Various pollution types
- Input size: 640x640 pixels
- Output: Bounding boxes with class labels and confidence scores

## Project Structure

```
AquaVisionAI/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── models/                 # Trained model files
│   ├── marine_species_model_90_perfect.h5
│   ├── coral_health_final_model.h5
│   └── pollution_detection.pt
├── src/                    # Source code
│   ├── main.py             # Core system implementation
│   ├── models/             # Model training scripts
│   │   ├── segmentation.py
│   │   ├── coral_reef.py
│   │   └── detection.py
│   └── preprocessing/      # Image preprocessing
│       └── image_enhancement.py
├── config/                 # Configuration files
│   └── detection.yaml      # YOLO model configuration
└── data/                   # Data directory
    ├── raw/                # Raw datasets
    └── processed/          # Processed datasets
```

## Future Improvements

- Add support for video analysis
- Implement real-time monitoring capabilities
- Expand species identification to include more marine life
- Add depth estimation for 3D reconstruction
- Implement cloud-based deployment for remote monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Marine biologists and researchers who provided expertise
- Open-source computer vision community
- Environmental conservation organizations

---

**AquaVisionAI** - Empowering marine conservation through AI
