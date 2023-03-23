from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import lit
from pyspark.ml.recommendation import ALS

def load_movie_names():
    movie_names = {}
    with open("/home/pramod4lk/Documents/pyspark-job-movie-lens/data/ml-100k/u.item", encoding = "ISO-8859-1") as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1]#.decode('ascii', 'ignore')
    return movie_names

def parse_input(line):
    fields = line.value.split()
    return Row(userID = int(fields[0]), movieID = int(fields[1]), rating = float(fields[2]))

if __name__ == "__main__":

    spark = SparkSession.builder.appName("MovieRecs").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    movie_names = load_movie_names()

    # Get raw data
    lines = spark.read.text("/home/pramod4lk/Documents/pyspark-job-movie-lens/data/ml-100k/u.data").rdd

    rating_rdd = lines.map(parse_input)

    ratings = spark.createDataFrame(rating_rdd).cache()

    als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="movieID", ratingCol="rating")
    model = als.fit(ratings)

    print("\nRatings for user ID 0:")
    user_ratings = ratings.filter("userID = 0")
    for rating in user_ratings.collect():
        print(movie_names[rating['movieID']], rating['rating'])

    print("\nTop 20 recommendations:")
    rating_counts = ratings.groupBy("movieID").count().filter("count > 100")
    popular_movies = rating_counts.select("movieID").withColumn("userID", lit(0))

    recommendations = model.transform(popular_movies)

    top_recommendations = recommendations.sort(recommendations.prediction.desc()).take(20)

    for recommendation in top_recommendations:
        print(movie_names[recommendation["movieID"]], recommendation["prediction"])

    spark.stop()