from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    # Method to transform data into vectors
    def vector_transform(self, X_train, X_test):
        X_train_tfidf = self.vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = self.vectorizer.transform(X_test).toarray()
        return X_train_tfidf, X_test_tfidf