**Project Documentation: Customer Segmentation and Purchase Prediction**

**Introduction:**
This project focuses on analyzing a customer dataset to gain insights and make predictions about customer behavior. It encompasses various data analysis tasks, including regression, classification, clustering, and deep learning-based sentiment analysis.

**Dependencies:**
The project requires the following libraries and tools to be installed:
- `pandas`: For data manipulation and preprocessing.
- `scikit-learn`: For machine learning tasks such as regression, classification, and clustering.
- `matplotlib`: For data visualization.
- `keras`: For building and training deep learning models.

**Code Overview:**
The provided code aims to achieve the project objectives through several defined functions and a main execution block.

**Constants:**
- `DATA_PATH`: The path to the CSV file containing the customer data. (the data used for this project is this database from kaggle. https://www.kaggle.com/datasets/vipin20/transaction-data)
- `NUM_CLUSTERS`: The number of clusters to use in the K-Means clustering.
- `MAX_SEQUENCE_LENGTH`: The maximum sequence length for text padding in sentiment analysis.
- `NUM_WORDS`: The vocabulary size for tokenization in sentiment analysis.
- `EMBEDDING_DIM`: The dimension of word embeddings in sentiment analysis.
- `NUM_EPOCHS`: The number of epochs for training the sentiment analysis model.
- `BATCH_SIZE`: The batch size for training the sentiment analysis model.

**Functions:**
1. `load_data(path)`: Loads the customer dataset from the specified path using `pandas`.

2. `preprocess_data(data)`: Cleans and preprocesses the loaded data. It fills missing values, calculates the 'TotalPurchaseAmount', and performs one-hot encoding on the 'Country' column.

3. `perform_regression(data)`: Performs linear regression to predict purchase amounts. Splits the data, trains the model, predicts on the test set, and calculates the mean squared error.

4. `perform_classification(data)`: Performs classification to identify high-value customers. Creates a new 'HighValueCustomer' column, splits the data, trains a RandomForest classifier, predicts on the test set, and prints accuracy and classification report.

5. `perform_clustering(data)`: Performs K-Means clustering to segment customers based on purchase behavior. Adds a 'Cluster' column to the data, visualizes the clusters using a scatter plot.

6. `perform_sentiment_analysis(data)`: Performs sentiment analysis on customer reviews using a deep learning model. Tokenizes and pads the text data, builds an LSTM-based model, compiles, and trains it.

7. `main()`: The main execution function. Loads the data, preprocesses it, and sequentially calls the defined analysis functions.

**Execution:**
To execute the project, ensure that the required libraries are installed. Adjust the `DATA_PATH` to point to the appropriate CSV file. Run the script, and it will perform the defined analysis steps on the provided customer dataset.

**Conclusion:**
This project demonstrates a comprehensive analysis of customer behavior using a variety of data analysis techniques. The code showcases how to predict purchase amounts, classify high-value customers, segment customers, and perform sentiment analysis on text data. The project provides valuable insights into customer preferences and allows for making informed business decisions.
