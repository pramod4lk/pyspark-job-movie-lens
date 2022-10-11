from pyspark.sql import SparkSession, Row, functions

def load_movie_names():
    movie_names = {}
    with open("ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1]
    return movie_names

def parse_input(line):
    fields = line.split()
    return Row(movieID = int(fields[1]), rating = float(fields[2]))

if __name__ == '__main__':

    spark = SparkSession.builder.appName('PopularMovie').getOrCreate()

    movie_names = load_movie_names()

    lines = spark.sparkContext.textFile('hdfs:///user/maria_dev/ml-100k/u.data')

    movies = lines.map(parse_input)

    movie_dataset = spark.createDataFrame(movies)

    average_ratings = movie_dataset.groupBy('movieID').avg('rating')

    counts = movie_dataset.groupBy('movieID').count()

    averges_and_counts = counts.join(average_ratings,'movieID')

    top_ten = averges_and_counts.orderBy('ang(rating)').take(10)

    for movie in top_ten:
        print (movie_names[movie[0]], movie[1], movie[2])

    spark.stop()