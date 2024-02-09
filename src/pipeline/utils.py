import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.svm import SVC
import pickle
import os

# Preprocess function
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmatized_tokens)

# Evaluate model metrics
def evaluate_model(true, predicted):
    ac_score = accuracy_score(true, predicted)
    pr_score = precision_score(true, predicted)
    con_matrix = confusion_matrix(true, predicted)
    return ac_score, pr_score, con_matrix

# Save object
def save_object(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)
        
# Print metrics
def print_metrics(accuracy, precision, c_matrix):
    print('Model performance')
    print('- Accuracy Score: {:.4f}'.format(accuracy))
    print('- Precision Score: {:.4f}'.format(precision))
    print('- Confusion Matrix:\n{}'.format(c_matrix))
    
# Load pickle
def load_pkl(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        obj = pickle.load(f)
        return obj
