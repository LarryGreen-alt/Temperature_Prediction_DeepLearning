# Temperature_Prediction_DeepLearning
This project investigates the use of deep learning techniques for weather forecasting by comparing two sequential learning architectures: Long Short-Term Memory (LSTM) networks and Transformer-based models.

Model 1: Long Short-Term Memory (LSTM) 

The first model will utilize PyTorch’s LSTM implementation to learn temporal dependencies within historical weather observations. 

Input Sequence à Normalization à LSTM Layer à Dropout à Dense Layer (ReLU) à Output Layer (Linear) à Temperature Prediction 

The LSTM model serves as a strong baseline for sequential forecasting tasks and has been extensively used in weather prediction research. 


Model 2: Transformer 

The second model will utilize a Transformer Encoder architecture implemented using PyTorch. 

Input Sequence à Positional Encoding à Transformer Encoder à Feed Forward Network à Output Layer (Linear) à Temperature Prediction 

Unlike LSTM networks, Transformers use self-attention mechanisms that allow the model to consider all previous observations simultaneously when generating predictions. This may enable the model to capture long-range temporal dependencies more effectively. Also, this presents a comparative analysis over the duration of our project between the two models. 

The proposed architectures represent the initial experimental design. During implementation, adjustments to the models and preprocessing pipeline may be introduced based on experimental results and computational considerations. Any such changes will be documented and justified in the final report. 
