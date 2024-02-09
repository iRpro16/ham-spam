from src.data.make_dataset import Dataset
from src.features.data_processor import DataProcessor
from src.features.vectorizer import Vectorizer
from src.models.train_model import Model
from src.pipeline.utils import evaluate_model, print_metrics

if __name__ == "__main__":
    # Get data
    data = Dataset('spam.csv')
    data.preprocess_text()

    # Split into X and y
    X, y = data.get_features()

    # Split into train and test
    split = DataProcessor()
    X_train, X_test, y_train, y_test = split.split_data(X, y)

    # Vectorize
    vectorize = Vectorizer()
    X_train_tfidf, X_test_tfidf = vectorize.vector_transform(X_train, X_test)

    # Model
    model = Model()
    model_fit = model.fit_model(X_train_tfidf, y_train)
    predict  = model_fit.predict(X_test_tfidf)
    
    # Store variables
    acc, pr, cm = evaluate_model(y_test, predict)
    
    #Print metrics
    print_metrics(acc, pr, cm)
   
    