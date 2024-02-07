from sklearn.model_selection import train_test_split

# Class to split data into features & targets
class DataProcessor:
    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)