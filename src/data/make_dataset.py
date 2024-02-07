import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.pipeline.utils import preprocess

class Dataset:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
    
    # Cleaning up data and preprocessing text
    def preprocess_text(self):
        self.df.drop_duplicates(keep='first', inplace=True)
        self.df['Category'] = self.label_encoder.fit_transform(self.df['Category'])
        # Creating new column and preprocessing for vectorization
        self.df['preprocessed_text'] = self.df['Message'].apply(preprocess).astype(str)