# In your training script (src/train.py)
from src.data_processing import create_data_processing_pipeline
import pandas as pd

# Load raw data
df = pd.read_excel('data/data.xlsx')
X = df.drop(columns=['FraudResult'])
y = df['FraudResult']

# Create and fit pipeline
pipeline = create_data_processing_pipeline()
X_processed = pipeline.fit_transform(X, y)

# X_processed is now model-ready!
print(X_processed.shape)
print(X_processed.columns.tolist())

# Save pipeline for later use
import joblib
joblib.dump(pipeline, 'models/data_pipeline.joblib')