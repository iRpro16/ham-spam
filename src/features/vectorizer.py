from sklearn.feature_extraction.text import TfidfVectorizer
from src.pipeline.utils import save_object
import numpy as np
import os
import pickle

class Vectorizer:
    def __init__(self):
        # File path to store vector.pkl
        self.vectorizer_config = os.path.join('models','vectorizer.pkl')
    
    def vector_transform(self, X_train, X_test):
        self.vectorizer = TfidfVectorizer()
        X_train_v = self.vectorizer.fit_transform(X_train)
        X_test_v = self.vectorizer.transform(X_test)
        
        # Save Vectorizer
        ''' 
        save_object(
            obj=self.vectorizer,
            file_path= self.vectorizer_config
        )
        '''
        
        return X_train_v, X_test_v
