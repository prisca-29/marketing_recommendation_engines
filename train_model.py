# train_model.py

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load dataset
data_path = "/content/mobile_recommendation_system_dataset.csv"
df = pd.read_csv(data_path)

# Inspect the dataset structure
print(df.head())

# Assuming dataset has 'user_id', 'item_id', and 'rating' columns
# Adjust the column names as per your dataset
df = df[['user_id', 'item_id', 'rating']]

# Use Surprise's Reader class to load data
reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
data = Dataset.load_from_df(df, reader)

# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD algorithm (you can replace this with other algorithms available in Surprise)
algo = SVD()

# Train the model
algo.fit(trainset)

# Make predictions on the test set
predictions = algo.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Save the trained model
import pickle
with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump(algo, f)

print("Model training complete and saved as recommendation_model.pkl")
