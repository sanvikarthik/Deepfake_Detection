# Deepfake_Detection
Deepfake detection using advanced AI techniques, including CNNs, multi-attention mechanisms, and landmark-based methods, for enhanced accuracy and robustness
# AI-Forensics-DeepFake-Detection

## 🌟 Project Overview

**AI-Forensics-DeepFake-Detection** is a robust and scalable pipeline designed to identify manipulated video content with precision. By leveraging cutting-edge techniques like spatial, temporal, frequency, and audio-visual analyses, it provides a comprehensive solution to detect deep fakes. The project integrates advanced deep learning models (CNNs, RNNs, GANs), Fourier analysis, and lip-sync verification, ensuring accuracy and reliability in identifying fake media.

---

## 📋 Table of Contents

- [✨ Features](#-features)  
- [🛠️ Technologies Used](#️-technologies-used)  
- [⚙️ Installation](#️-installation)  
- [🌀 Pipeline Overview](#-pipeline-overview)  
- [🚀 Usage](#-usage)  
- [📈 Future Work](#-future-work)  
- [🤝 Contributing](#-contributing)  
- [📜 License](#-license)  

---

## ✨ Features

- **Frame Analysis**: Extracts video frames and detects faces for consistent alignment and tracking.  
- **Spatial Detection**: Leverages CNN models to analyze texture and facial features for abnormalities.  
- **Temporal Detection**: Uses LSTM to identify unnatural transitions between frames.  
- **Frequency Analysis**: Applies Fourier transforms to detect frequency-domain artifacts.  
- **Audio-Visual Sync**: Integrates SyncNet to detect lip-sync mismatches in manipulated videos.  
- **Detailed Reporting**: Generates confidence scores and summaries of detected inconsistencies.  

---

## 🛠️ Technologies Used

### **Machine Learning and Deep Learning**
- **TensorFlow** / **PyTorch**: For building CNNs, RNNs, GANs, and other models.  
- **Keras**: High-level deep learning API for quick prototyping.  
- **Scikit-learn**: Supplemental algorithms and evaluation metrics.  

### **Audio-Visual Analysis**
- **SyncNet**: Detects lip-sync mismatches to verify audio-visual coherence.  

### **Frequency Domain Analysis**
- **SciPy**: Implements Fourier and Wavelet transforms for detecting artifacts in frequency domains.  

### **Frontend Development**
- **Streamlit**: Creates an interactive and user-friendly interface for seamless operation.  

---

## ⚙️ Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/AI-Forensics-DeepFake-Detection.git
   cd AI-Forensics-DeepFake-Detection
   Install Dependencies
2. Make sure to have Python 3.8+ installed, then run:

```bash
  #pip install -r requirements.txt
```
3. Run the Application
  Launch the GUI:
```
streamlit run myapp.py
 ```
4. Verify Setup
Ensure all files are in place and dependencies are correctly installed.

## Pipeline Overview
Step 1: Extract frames from the input video and preprocess them for analysis.
Step 2: Perform spatial, temporal, and frequency analysis on the frames.
Step 3: Conduct audio-visual synchronization checks using SyncNet.
Step 4: Aggregate results and generate a detailed report.

## 🚀 Usage
Upload the video file you want to analyze.
Run the detection pipeline via the user interface.
Review the detailed report highlighting detected inconsistencies.

## 📈 Future Work
Streamlit UI Enhancements: Revamp the UI for better usability and visualization.
Extended Model Training: Incorporate more datasets for improved accuracy.
Missing File Check: Add functionality to flag missing or corrupted files automatically.
Additional Features: Explore integrating other modalities like metadata analysis.

## 🤝 Contributing
Contributions are welcome! If you have ideas, improvements, or find bugs, please open an issue or submit a pull request.



📝 To-Do List
Migrate the UI to Streamlit.
Check for missing or incomplete files.
Ensure the logic within myapp.py remains functional when making UI changes.
Suggest additional improvements if necessary.
Feel free to reach out for any clarifications or feature requests! 🚀
