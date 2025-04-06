Audio Deepfake Detection using RawNet2 with Light Fine-Tuning
Overview
This project implements the RawNet2 model for the task of audio deepfake (spoof) detection. The goal is to identify whether a given speech sample is bonafide (genuine) or spoofed (AI-generated or manipulated). A small subset of the ASVspoof dataset was used due to storage limitations. The model was lightly fine-tuned by freezing all layers except the final classification layer.

Model
Model: RawNet2 (pretrained)

Approach:

All layers were frozen during training except the fully connected classification layer.

The model was trained using a balanced subset: 85 bonafide and 85 spoofed samples.

Training was conducted over 5 epochs to demonstrate basic fine-tuning capability.

Dataset
Source: ASVspoof Challenge dataset

Subset Used: Small balanced subset (85 bonafide + 85 spoof)

Format: FLAC audio files

Reason for Subsetting: Due to storage limitations in the working environment, a reduced subset was used for demonstration purposes.

Performance Results
Epoch	Loss	Accuracy
1	23.8598	44.85%
2	23.5800	50.00%
3	23.5141	50.74%
4	23.4110	51.47%
5	23.2845	52.94%
Note: Performance is not the primary metric here. The goal was to demonstrate the model’s integration and fine-tuning pipeline.

Setup Instructions
Requirements
Install the required packages:

bash
Copy
Edit
pip install torch torchaudio numpy matplotlib
Running the Code
Upload the Jupyter Notebook (.ipynb) to Google Colab or run it in a local Jupyter environment.

Mount Google Drive (if using Colab) to access dataset files.

Execute cells in order for data loading, model training, and evaluation.

File Structure
bash
Copy
Edit
├── rawnet2_finetune.ipynb     # Main implementation notebook
├── rawnet2_trained.pth        # Pretrained model weights (if included)
├── README.md                  # Project overview and instructions
Notes
The training used only a small sample size due to limited storage availability.

This work focuses on demonstrating a functional pipeline and basic fine-tuning approach for RawNet2.

Future work can include training on the full ASVspoof dataset or experimenting with other models such as Wav2Vec2 or LCNN.
