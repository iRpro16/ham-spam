from sklearn.feature_extraction.text import TfidfVectorizer
from src.pipeline.utils import save_object
import numpy as np
import os
import pickle

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer_config = os.path.join('models','vectorizer.pkl')
        
        #Save Vectorizer
        ''' 
        save_object(
            obj=self.vectorizer,
            file_path= self.vectorizer_config
        )
        '''
    
    # Method to transform data into vectors
    def vector_transform(self, X_train, X_test):
        X_train_tfidf = self.vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = self.vectorizer.transform(X_test).toarray()
        return X_train_tfidf, X_test_tfidf
    
