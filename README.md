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

**Installation**

Clone the repository:

git clone https://github.com/<username>/WeatherAI.git

Create a virtual environment:

python -m venv venv

Activate it.

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

**Features**
- Historical weather data collection
- Automated preprocessing pipeline
- Train / Validation / Test dataset generation
- LSTM forecasting model
- Transformer forecasting model
- Model checkpointing
- Prediction generation
- Training history logging
- Evaluation metrics
- Automatic graph generation
- Modular project architecture

**Structure**
'''
```text
WeatherAI/
│
├── .env
├── .gitignore
├── README.md
├── weather_main.py
│
├── data/
│   ├── processed/
│   │   └── features.csv
│   │
│   ├── raw/
│   │   ├── atlanta_2024.csv
│   │   ├── austin.csv
│   │   ├── boston.csv
│   │   ├── dallas.csv
│   │   ├── los_angeles.csv
│   │   ├── random.csv
│   │   ├── san_antonio.csv
│   │   ├── seattle.csv
│   │   └── seattle_2024.csv
│   │
│   └── splits/
│       ├── train.csv
│       ├── dev.csv
│       └── test.csv
│
├── models/
│   ├── weather_lstm.keras
│   │
│   ├── history/
│   │   ├── metrics.json
│   │   ├── predictions.csv
│   │   └── LSTM/
│   │       └── training_history.csv
│   │
│   ├── LSTM/
│   │   ├── checkpoints/
│   │   │   └── best.keras
│   │   │
│   │   └── experiments/
│   │       ├── 2026-06-30_17-20-16/
│   │       │   ├── model_summary.txt
│   │       │   ├── training_history.csv
│   │       │   ├── weather_transformer.keras
│   │       │   └── figures/
│   │       │       ├── loss_curve.png
│   │       │       ├── mae_curve.png
│   │       │       └── prediction_curve.png
│   │       │
│   │       ├── 2026-06-30_18-37-23/
│   │       │   ├── model_summary.txt
│   │       │   ├── training_history.csv
│   │       │   ├── weather_lstm.keras
│   │       │   └── figures/
│   │       │       ├── loss_curve.png
│   │       │       ├── mae_curve.png
│   │       │       └── prediction_curve.png
│   │       │
│   │       └── 2026-07-01_17-14-36/
│   │           ├── metrics.json
│   │           ├── model_summary.txt
│   │           ├── predictions.csv
│   │           ├── training_history.csv
│   │           ├── weather_lstm.keras
│   │           └── figures/
│   │               ├── loss_curve.png
│   │               ├── mae_curve.png
│   │               └── prediction_curve.png
│   │
│   └── predictions/
│       └── predictions.csv
│
├── src/
│   ├── api/
│   │   └── openmeteo_client.py
│   │
│   ├── data/
│   │   ├── collect_historical.py
│   │   ├── preprocess.py
│   │   └── split_dataset.py
│   │
│   ├── models/
│   │   ├── checkpoints/
│   │   │   └── Transformer/
│   │   │
│   │   ├── common/
│   │   │   ├── callbacks.py
│   │   │   ├── evaluation.py
│   │   │   └── plotting.py
│   │   │
│   │   ├── lstm/
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   └── predict.py
│   │   │
│   │   └── transformer/
│   │       ├── model.py
│   │       ├── train.py
│   │       └── predict.py
│   │
│   └── utils/
│       ├── city_coordinates.py
│       └── config.py
│
└── tests/
```

'''

**Machine Learning Pipeline**
Historical Weather Data
            │
            ▼
     Data Collection
            │
            ▼
      Preprocessing
            │
            ▼
    Feature Engineering
            │
            ▼
Train / Validation / Test Split
            │
            ▼
      Model Training
            │
            ▼
        Evaluation
            │
            ▼
        Prediction

**Data Preparation**

Collect historical weather data:

python -m src.data.collect_historical

Preprocess:

python -m src.data.preprocess

Create dataset splits:

python -m src.data.split_dataset

**Training**

Train the LSTM model:

python -m src.models.lstm.train

Train the Transformer model:

python -m src.models.transformer.train

**Making Predictions**

LSTM:

python -m src.models.lstm.predict

Transformer:

python -m src.models.transformer.predict

Predictions are saved to

models/predictions/

**Generated Outputs**

Each experiment automatically saves:

Trained model
Model summary
Training history
Prediction CSV
Evaluation metrics
Loss curve
MAE curve
Prediction curve
