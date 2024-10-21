# Install the surprise library for building recommendation systems
# !pip install scikit-surprise

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

# Step 1: Load the dataset
# Let's create a simple dataset where users rate songs (user_id, song_id, rating)
import pandas as pd

# Example data: User ratings for different songs (1-5 scale)
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'song_id': [101, 102, 103, 101, 102, 104, 103, 104, 105],
    'rating':  [5, 3, 4, 4, 2, 5, 3, 4, 2]
}

df = pd.DataFrame(data)

# Step 2: Prepare the data for the Surprise library
# The Reader class allows you to specify the format of the dataset (rating scale is 1-5)
reader = Reader(rating_scale=(1, 5))

# Load the data from the dataframe
dataset = Dataset.load_from_df(df[['user_id', 'song_id', 'rating']], reader)

# Step 3: Split the data into training and test sets
trainset, testset = train_test_split(dataset, test_size=0.25)

# Step 4: Build the recommendation model (using SVD)
model = SVD()

# Step 5: Train the model on the training set
model.fit(trainset)

# Step 6: Evaluate the model using cross-validation
# This evaluates the model using the RMSE (Root Mean Squared Error)
cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Step 7: Make predictions for a specific user and song
# Predict the rating for user 2 for song 103
user_id = 2
song_id = 103
pred = model.predict(user_id, song_id)
print(f"Predicted rating for user {user_id} and song {song_id}: {pred.est:.2f}")

# Step 8: Recommend top songs for a specific user (e.g., user 2)
# Let's recommend 3 songs for user 2
user_id = 2

# Get a list of all song ids from the dataset
song_ids = df['song_id'].unique()

# Predict ratings for all songs the user hasn't rated yet
songs_not_rated_by_user = [song for song in song_ids if not df[(df['user_id'] == user_id) & (df['song_id'] == song)].any().any()]

# Predict ratings for those songs
predictions = [model.predict(user_id, song_id) for song_id in songs_not_rated_by_user]

# Sort the predictions by the estimated rating
predictions.sort(key=lambda x: x.est, reverse=True)

# Print the top 3 recommended songs for the user
print(f"\nTop 3 recommended songs for user {user_id}:")
for pred in predictions[:3]:
    print(f"Song ID: {pred.iid}, Predicted Rating: {pred.est:.2f}")
