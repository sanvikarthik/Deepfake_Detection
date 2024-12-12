# Deepfake_Detection
Deepfake detection using advanced AI techniques, including CNNs, to ensure high accuracy and scalability for real-world applications.

# AI-Forensics-DeepFake-Detection

## ✨ Project Overview

**AI-Forensics-DeepFake-Detection** is an advanced pipeline developed to identify manipulated media with precision and efficiency. This project focuses on leveraging Convolutional Neural Networks (CNNs) and optimized architectures like EfficientNetAutoAttB4 to detect subtle artifacts introduced during deepfake creation, such as lighting mismatches, texture irregularities, and temporal inconsistencies. By focusing on accuracy, scalability, and real-time processing, this project provides a practical solution for combating deepfake threats.

---

## 📜 Table of Contents

- [✨ Features](#-features)  
- [🔧 Technologies Used](#-technologies-used)  
- [⚙️ Installation](#-installation)  
- [🔄 Pipeline Overview](#-pipeline-overview)  
- [🚀 Usage](#-usage)  
- [📊 Future Work](#-future-work)  
- [🤝 Contributing](#-contributing)  
- [📌 License](#-license)  

---

## ✨ Features

- **Frame Analysis**: Extracts frames from videos for preprocessing and analysis.  
- **Spatial Detection**: Uses CNN-based models to detect texture and facial inconsistencies.  
- **Artifact Analysis**: Identifies lighting mismatches and texture irregularities indicative of deepfake manipulation.  
- **Temporal Analysis**: Analyzes consecutive frames to detect unnatural transitions.  
- **Detailed Reporting**: Provides confidence scores and highlights manipulated areas for easy interpretation.  

---

## 🔧 Technologies Used

### **Machine Learning and Deep Learning**
- **EfficientNetAutoAttB4**: Optimized CNN architecture for visual data analysis.  
- **TensorFlow** / **PyTorch**: Frameworks used for model implementation and training.  

### **Frontend Development**
- **Streamlit**: Provides an intuitive user interface for video uploads and results visualization.  

---

## ⚙️ Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/sanvikarthik/Deepfake_Detection
   cd AI-Forensics-DeepFake-Detection
   ```

2. **Install Dependencies**  
   Ensure Python 3.8+ is installed, then run:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**  
   Launch the Streamlit UI:  
   ```bash
   streamlit run app.py
   ```

4. **Verify Setup**  
   Check that all files and dependencies are correctly installed and functional.

---

## 🔄 Pipeline Overview

1. **Data Preprocessing**: Extracts frames from input videos, resizes them, and normalizes them for model input.  
2. **Spatial and Temporal Analysis**: Detects inconsistencies in texture and transitions between frames using CNNs.  
3. **Artifact Detection**: Identifies subtle anomalies such as lighting mismatches and texture irregularities.  
4. **Report Generation**: Outputs confidence scores and visual highlights of detected manipulations.  

---

## 🚀 Usage

1. Upload the video file to the Streamlit interface.
2. Run the detection pipeline.
3. Review the generated report, including highlighted inconsistencies and confidence scores.

---

## 📊 Future Work

- **Enhanced Dataset Integration**: Incorporate additional datasets to improve model robustness.  
- **UI Enhancements**: Revamp the Streamlit interface for better visualization and user experience.  
- **Expanded Modalities**: Explore integration of audio and metadata analysis for deeper insights.  
- **Real-Time Detection**: Optimize the pipeline for real-time processing on video streams.  

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or discover any bugs, feel free to open an issue or submit a pull request.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute with attribution.

---

## 🖋️ To-Do List

- Ensure Streamlit UI is functional and user-friendly.  
- Validate model accuracy with additional testing.  
- Incorporate real-time detection capabilities for live video feeds.  
- Explore further integration of other AI-based solutions for enhanced detection.
