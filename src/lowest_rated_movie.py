from dataclasses import fields
from json import load
from unittest import result
from pyspark import SparkConf, SparkContext
from dataclasses import fields, field

def load_movie_names():
    movie_names = {}
    with open("data/ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1]
    return movie_names

def parse_input(line):
    fields = line.split()
    return (int(fields[1]), (float(fields[2]), 1.0))

if __name__ == '__main__':
    conf = SparkConf().setAppName("WorstMovies")
    sc = SparkContext(conf=conf)

    movie_names = load_movie_names()

    lines = sc.textFile('hdfs:///user/maria_dev/sampledata/ml-100k/u.data')

    movie_ratings = lines.map(parse_input)

    rating_totals_and_count = movie_ratings.reduceByKey(lambda movie1, movie2: (movie1[0] + movie2[0]))

    average_ratings = rating_totals_and_count.mapValues(lambda total_and_count: total_and_count[0] / total_and_count[1])

    sorted_movies = average_ratings.sortBy(lambda x:x[1])

    results = sorted_movies.take(10)

    for result in results:
        print(movie_names[result[0]], result[1])