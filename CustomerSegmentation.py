import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Constants
DATA_PATH = 'C:\\Users\\pc\\Downloads\\transaction_data.csv'
NUM_CLUSTERS = 4
MAX_SEQUENCE_LENGTH = 100
NUM_WORDS = 36957  # Define the vocabulary size
EMBEDDING_DIM = 200  # Define the embedding dimension
NUM_EPOCHS = 20  # Define the number of epochs
BATCH_SIZE = 64  # Define the batch size

# Load the dataset
def load_data(path):
    return pd.read_csv(path)

# Preprocess the dataset
def preprocess_data(data):
    data.fillna(0, inplace=True)
    data['TotalPurchaseAmount'] = data['NumberOfItemsPurchased'] * data['CostPerItem']
    data = pd.get_dummies(data, columns=['Country'], drop_first=True)
    return data

# Perform linear regression on purchase data
def perform_regression(data):
    X = data[['NumberOfItemsPurchased', 'CostPerItem']]
    y = data['TotalPurchaseAmount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (Regression): {mse}")

# Perform classification to identify high-value customers
def perform_classification(data):
    data['HighValueCustomer'] = (data['TotalPurchaseAmount'] > 1000).astype(int)
    X_class = data[['NumberOfItemsPurchased', 'CostPerItem']]
    y_class = data['HighValueCustomer']
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    class_model = RandomForestClassifier()
    class_model.fit(X_train_class, y_train_class)
    y_pred_class = class_model.predict(X_test_class)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    class_rep = classification_report(y_test_class, y_pred_class)
    print(f"Accuracy (Classification): {accuracy}")
    print(class_rep)

# Perform clustering to segment customers
def perform_clustering(data):
    X_cluster = data[['NumberOfItemsPurchased', 'CostPerItem']]
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_cluster)
    plt.scatter(data['NumberOfItemsPurchased'], data['CostPerItem'], c=data['Cluster'], cmap='rainbow')
    plt.xlabel('Number of Items Purchased')
    plt.ylabel('Cost Per Item')
    plt.title('Customer Segmentation')
    plt.show()

# Perform sentiment analysis using deep learning
def perform_sentiment_analysis(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['CustomerReview'])
    X_text = tokenizer.texts_to_sequences(data['CustomerReview'])
    X_text = pad_sequences(X_text, maxlen=MAX_SEQUENCE_LENGTH)
    model = Sequential()
    model.add(Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_text, data['SentimentLabel'], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# Main function to execute the project tasks
def main():
    data = load_data(DATA_PATH)
    data = preprocess_data(data)
    
    perform_regression(data)
    perform_classification(data)
    perform_clustering(data)
    perform_sentiment_analysis(data)

if __name__ == '__main__':
    main()
