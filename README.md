# Movies_recommendations
 
Collaborative Filtering Movie Recommendation System using PySpark

This repository contains a collaborative filtering movie recommendation system implemented in PySpark. The system is built using the Alternating Least Squares (ALS) algorithm for matrix factorization.

Files and Data

- `Movies_suggestions_10M.py`: Using dataset ml-10M100K
- `Movies_suggestions_25M.py`: Using dataset ml-25m (find it here: https://grouplens.org/datasets/movielens/25m/)
- `ml-10M100K/ratings.dat`: Movie ratings dataset in the format (UserID, MovieID, Rating).
- `ml-10M100K/movies.dat`: Movies dataset in the format (MovieID, Title).

Usage

**Install Dependencies:**
   Make sure you have PySpark installed on your system. You can install it using the following command:
   ```bash
   pip install pyspark
   ```
   
**Provide Ratings:**
   Follow the prompts to rate a selection of movies. Enter ratings between 1 and 5, or 0 if you don't know. This input is used to generate personalized recommendations.

**Results:**
   The script will output the Mean Squared Error (MSE) of the recommendation model on the test data and provide the top-k movie recommendations for a specified user.

## Additional Information

- The script randomly selects 40 movies from the 200 most-rated movies for user ratings elicitation.
- The recommendation model is trained using ALS with parameters: rank=8, numIterations=20.

Feel free to explore and modify the script according to your needs. If you encounter any issues or have suggestions, please create an issue in the repository.

Happy movie recommending!
