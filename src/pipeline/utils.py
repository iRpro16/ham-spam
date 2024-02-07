import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

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