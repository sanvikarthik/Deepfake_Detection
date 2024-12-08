# AI-Forensics-DeepFake-Detection


## Project Overview

AI-Forensics-DeepFake-Detection is a comprehensive deep fake detection pipeline. It uses a combination of spatial, temporal, frequency, and audio-visual analyses to detect manipulated videos. This project combines deep learning models (CNN, RNN, GANs), Fourier analysis, and lip-sync verification, providing an accurate, scalable solution for identifying fake media content. 

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Frame Analysis**: Extracts frames and detects faces for consistent alignment and tracking.
- **Spatial Detection**: Uses a CNN model to analyze texture and facial details for inconsistencies.
- **Temporal Detection**: Employs LSTM to detect unnatural transitions across frames.
- **Frequency Analysis**: Uses Fourier transforms to identify frequency-domain artifacts.
- **Audio-Visual Sync**: SyncNet checks for lip-sync mismatches.
- **Detailed Reporting**: Generates a comprehensive report with confidence scores for each type of inconsistency.

---

## Technologies Used

- **Machine Learning and Deep Learning**:
  - `TensorFlow` or `PyTorch`: For CNN, RNN, LSTM, and GAN model building.
  - `Keras`: High-level API to quickly prototype deep learning models.
  - `Scikit-learn`: Additional algorithms and metrics.
  
- **Audio-Visual Analysis**:
  - `SyncNet`: Lip-sync detection for audio-visual coherence.

- **Frequency Domain Analysis**:
  - `SciPy`: Fourier and Wavelet transforms for artifact detection.

- **Frontend Development**:
  - `Streamlit`: For developing an interactive, user-friendly interface.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/AI-Forensics-DeepFake-Detection.git
   cd AI-Forensics-DeepFake-Detection
   
---

## To Do

1. **Change the UI to streamlit**
2. **let me know if some files are missing**
3. **run myapp.py (contains logic and GUI elements if youre gonna change it take care of the logic)**
4. **Thats all i can think for now add something to this if theres anything else**
