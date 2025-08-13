# Deep learning project to classify Quran reciters from audio recordings using deep learning.
# Description
This project uses a Convolutional Neural Network (CNN) to identify different Quran reciters based on their audio recitations. The model converts audio files into mel-spectrograms and classifies them to predict which reciter is speaking.
Requirements
# Install the required packages:
pip install torch librosa scikit-learn matplotlib pandas numpy scikit-image torchsummary seaborn opendatasets
" Dataset
The project uses the "Quran Recitations for Audio Classification" dataset from Kaggle. The dataset is automatically downloaded when you run the code.
# How it Works

Audio Processing: Converts 5-second audio clips to mel-spectrograms
Model Training: Uses a CNN with 3 convolutional layers and 4 dense layers
Classification: Predicts which reciter is speaking with confidence scores

# Model Architecture

Input: Mel-spectrogram (128x256)
3 Convolutional layers (16, 32, 64 filters)
MaxPooling after each conv layer
4 Dense layers (4096, 1024, 512, output)
Dropout for regularization
CrossEntropy loss with Adam optimizer

# Training Settings

Learning Rate: 0.0001
Batch Size: 16
Epochs: 25
Train/Val/Test Split: 70%/15%/15%

# Results
The model provides:

Classification accuracy on test data
Confusion matrix showing per-class performance
Training/validation loss curves
Confidence scores for predictions