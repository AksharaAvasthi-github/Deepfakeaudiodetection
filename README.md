# Deepfakeaudiodetection

Project Overview
This project explores the detection of AI-generated (spoofed) human speech using deep learning techniques. The core implementation focuses on the RawNet2 model—an end-to-end convolutional neural network designed to classify raw audio waveforms as either bonafide (genuine) or spoof (fake). The ASVspoof dataset was used for training and evaluation, and light fine-tuning was performed by unfreezing only the final classification layer while keeping the rest of the model frozen. The goal was to demonstrate a working pipeline for deepfake audio detection, analyze model behavior, and understand its applicability in real-world scenarios.
