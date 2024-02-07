import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

class Dataset:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
    
    def df_clean(self):
        self.df.drop_duplicates(keep='first', inplace=True)
        self.df['Category'] = self.label_encoder.fit_transform(self.df['Category'])

