from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkConf, SparkContext
import random
import os

# Set up SparkConf
conf = SparkConf().setAppName("gwen").setMaster("local[8]")\
                  .set("spark.executor.memory", "4g")\
                  .set("spark.driver.memory", "4g")\
                  .set("spark.executor.memoryFraction", "0.8")



# Initialize SparkContext
sc = SparkContext(conf=conf)

# Load and parse the ratings data
data = sc.textFile("ml-25m/ratings.csv")
header_ratings = data.first()
ratings_data = data.filter(lambda line: line != header_ratings)
ratings = ratings_data.map(lambda l: l.split(','))\
                     .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Load and parse movies data
movies_data = sc.textFile("ml-25m/movies.csv")
header_movies = movies_data.first()
movies_data = movies_data.filter(lambda line: line != header_movies)
movies = movies_data.map(lambda l: l.split(','))\
                   .map(lambda l: (int(l[0]), str(l[1])))

# Find 200 most rated movies
# mostRatedMovies
movie_ratings_count = ratings.map(lambda r: (r.product, 1)).reduceByKey(lambda x, y: x + y)
top200_movie = movie_ratings_count.sortBy(lambda r: r[1], ascending=False).take(200)
mostRatedMovies = []

for movie_id, count in top200_movie:
    movie_title = movies.lookup(movie_id)
    if movie_title:
        mostRatedMovies.append((movie_id, movie_title[0]))

# Random shuffle
selectedMovies = random.sample(mostRatedMovies, 20)

def elicitateRatings(selectedMovies):
    user_ratings = []

    for movie_id, title in selectedMovies:
        # Get user input for the rating
        while True:
            try:
                rating = int(input(f"Rate the movie '{title}' (1-5) or enter 0 if you don't know: "))
                if 0 <= rating <= 5:
                    break
                else:
                    print("Please enter a valid rating between 0 and 5.")
            except ValueError:
                print("Please enter a valid integer.")

        # Append the movie rating to the user_ratings list
        user_ratings.append(Rating(88888, movie_id, rating))

    return user_ratings

user_ratings = elicitateRatings(selectedMovies)
sc_users_ratings = sc.parallelize(user_ratings)



# Combine the new_ratings_rdd with the existing 'ratings' RDD
ratings = ratings.union(sc_users_ratings)


#Split test/train
train, test = ratings.randomSplit(weights=[0.9,0.1], seed=100)


# Build the recommendation model using Alternating Least Squares
rank = 8
numIterations = 20
model = ALS.train(train, rank, numIterations)

# Evaluate the model on training data
testdata = test.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()

print("Mean Squared Error = " + str(MSE))




# Replace 'yourUserID' with the actual user ID you assigned to yourself
yourUserID = 88888

# Set the number of recommendations you want (k)
k = 10

# Get top-k movie recommendations for yourself
top_recommendations=[]
top_recommendations = model.recommendProducts(yourUserID, k)

os.system('clear')

for recommendation in top_recommendations:
    movie_id = recommendation.product
    movie_title = movies.lookup(movie_id)
    print(f"Movie ID: {movie_id}, Title: {movie_title}, Predicted Rating: {recommendation.rating:.2f}")


sc.stop()
