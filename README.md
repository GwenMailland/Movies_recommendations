#Movies_recommendations
 
Collaborative Filtering Movie Recommendation System using PySpark

This repository contains a collaborative filtering movie recommendation system implemented in PySpark. The system is built using the Alternating Least Squares (ALS) algorithm for matrix factorization.

Files and Data

- `recommendation_system.py`: PySpark script implementing the recommendation system.
- `ml-10M100K/ratings.dat`: Movie ratings dataset in the format (UserID, MovieID, Rating).
- `ml-10M100K/movies.dat`: Movies dataset in the format (MovieID, Title).

Usage

1. **Install Dependencies:**
   Make sure you have PySpark installed on your system. You can install it using the following command:
   ```bash
   pip install pyspark
   ```

2. **Run the Recommendation System:**
   Execute the `recommendation_system.py` script to generate movie recommendations. You can run the script using the following command:
   ```bash
   spark-submit recommendation_system.py
   ```

3. **Provide Ratings:**
   Follow the prompts to rate a selection of movies. Enter ratings between 1 and 5, or 0 if you don't know. This input is used to generate personalized recommendations.

4. **Results:**
   The script will output the Mean Squared Error (MSE) of the recommendation model on the test data and provide the top-k movie recommendations for a specified user.

## Additional Information

- The script randomly selects 40 movies from the 200 most-rated movies for user ratings elicitation.
- The recommendation model is trained using ALS with parameters: rank=8, numIterations=20.

Feel free to explore and modify the script according to your needs. If you encounter any issues or have suggestions, please create an issue in the repository.

Happy movie recommending!
