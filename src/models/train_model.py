from sklearn.svm import SVC
from src.pipeline.utils import save_object
import os

class Model:
    def __init__(self):
        # Path for to save model.pkl
        self.model_trainer_config = os.path.join("models","model.pkl")
        
    def fit_model(self, x_train_v, y_train):
        self.model = SVC()
        self.model.fit(x_train_v, y_train)
        
        # Save model
        ''' 
        save_object(
            obj=self.model,
            file_path=self.model_trainer_config
        )
        '''
    
        return self.model
    