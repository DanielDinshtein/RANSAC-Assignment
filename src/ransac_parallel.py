import random
import pandas as pd

import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

from src.utils import init_spark


# =========    Parallel     ============


def extract_data(spark, file_path):
    """
    Read samples from a csv file into Spark DataFrame.
    Clean unnecessary columns.
    Each sample contain 'x' and 'y'.

    :param spark: Spark session
    :param file_path: Path to the csv file - the Dataset
    :return: The Samples - type: Spark DataFrame
    """

    samplesSchema = StructType([
        StructField('_c0', IntegerType()),
        StructField('x', DoubleType()),
        StructField('y', DoubleType()),
    ])

    df = spark.read.format("csv") \
        .option("header", True) \
        .schema(samplesSchema) \
        .load(file_path) \
        .drop("_c0")

    return df


def init_models_DF(spark):
    """
    Create Models empty template Spark DataFrame

    :param spark: Spark session
    :return: Empty Models Spark DF
    """
    modelsSchema = StructType([
        StructField('x1', DoubleType()), StructField('y1', DoubleType()),
        StructField('x2', DoubleType()), StructField('y2', DoubleType()),
        StructField('a', DoubleType()), StructField('b', DoubleType())
    ])

    df = spark.createDataFrame([], schema=modelsSchema)

    return df


def get_random_sample_pairs(samples, num_of_pairs):
    """
    Picks pairs of random samples from the DataFrame of samples given.
    * It also makes sure they do not have the same x.

    :param samples: The Spark DataFrame of samples
    :param num_of_pairs: Number of pairs needed (number of iterations)
    :return: List of Pairs of samples as dictionary - {x1,y1,x2,y2}
    """

    pairs_to_add = num_of_pairs
    random_sample_pairs_lst = []

    while pairs_to_add != 0:
        random_samples_1 = samples.rdd.takeSample(True, pairs_to_add)
        random_samples_2 = samples.rdd.takeSample(True, pairs_to_add)

        for row_idx in range(pairs_to_add):

            print("1")

            if random_samples_1[row_idx]['x'] - random_samples_2[row_idx]['x'] != 0:
                random_sample_pairs_lst.append({
                    "x1": random_samples_1[row_idx]['x'],
                    "y1": random_samples_1[row_idx]['y'],
                    "x2": random_samples_2[row_idx]['x'],
                    "y2": random_samples_2[row_idx]['y'],
                })

        pairs_to_add = num_of_pairs - len(random_sample_pairs_lst)

    return random_sample_pairs_lst


def modelFromSamplePair(sample1, sample2):
    dx = sample1['x'] - sample2['x']
    if dx == 0:  # avoid division by zero later
        dx = 0.0001

    # model = <a,b> where y = ax+b
    # so given x1,y1 and x2,y2 =>
    #  y1 = a*x1 + b
    #  y2 = a*x2 + b
    #  y1-y2 = a*(x1 - x2) ==>  a = (y1-y2)/(x1-x2)
    #  b = y1 - a*x1

    a = (sample1['y'] - sample2['y']) / dx
    b = sample1['y'] - sample1['x'] * a
    return {'a': a, 'b': b}


def scoreModelAgainstSamples(model, samples, cutoff_dist=20):
    # predict the y using the model and x samples, per sample, and sum the abs of the distances between the real y
    # with truncation of the error at distance cutoff_dist

    totalScore = samples.withColumn('pred_y', model['a'] * samples['x'] + model['b']) \
        .withColumn('distance', F.abs(samples['y'] - F.col('pred_y'))) \
        .withColumn('score', F
                    .when(F.col('distance') <= F.lit(cutoff_dist), F.col('distance'))
                    .otherwise(F.lit(cutoff_dist)))

    # .otherwise(F.lit(cutoff_dist))).select(F.sum('score')).collect()[0][0]

    totalScore = totalScore.select('score').toPandas()['score'].sum()

    #  TODO: Remove
    # samples.printSchema()
    # samples.show(truncate=False)

    return totalScore


def parallel_ransac(file_path, iterations, cutoff_dist):
    spark, sc = init_spark()

    samples_df = extract_data(spark, file_path)

    models_df = init_models_DF(spark)

    random_sample_pairs = get_random_sample_pairs(samples_df, iterations)

    #  TODO:  Need This? & Remove
    # samples_df.persist()
    # samples_df.printSchema()
    # samples_df.show(truncate=False)

    min_m = {}
    min_score = -1
    # row_list = samples_df.collect()
    # for i in range(1, iterations):
    #     if i % 10 == 0:
    #         print(i)
    #     # sample1, sample2 = get_random_sample_pair(samples_df)
    #     # sample1, sample2 = get_random_sample_pair(row_list)
    #     m = modelFromSamplePair(sample1, sample2)
    #     score = scoreModelAgainstSamples(m, samples_df, cutoff_dist)
    #
    #     if min_score < 0 or score < min_score:
    #         min_score = score
    #         min_m = m
    #
    # print({'model': min_m, 'score': min_score})

    return {'model': min_m, 'score': min_score}


"""# =========    Serially     ============"""


# function that picks a pair of random samples from the list of samples given (it also makes sure they do not have the same x)


# def get_random_sample_pair(samples):
#     dx = 0
#     selected_samples = []
#     while dx == 0:
#         # keep going until we get a pair with dx != 0
#         selected_samples = []
#         for i in [0, 1]:
#             index = random.randint(0, len(samples) - 1)
#             x = samples[index]['x']
#             y = samples[index]['y']
#             selected_samples.append({'x': x, 'y': y})
#             # print("creator_samples ",i, " : ", creator_samples, " index ", index)
#         dx = selected_samples[0]['x'] - selected_samples[1]['x']
#
#     return selected_samples[0], selected_samples[1]


# generate a line model (a,b) from a pair of (x,y) samples
# sample1,2 -  is a dictionary with x and y keys
# def modelFromSamplePair(sample1, sample2):
#     dx = sample1['x'] - sample2['x']
#     if dx == 0:  # avoid division by zero later
#         dx = 0.0001
#
#     # model = <a,b> where y = ax+b
#     # so given x1,y1 and x2,y2 =>
#     #  y1 = a*x1 + b
#     #  y2 = a*x2 + b
#     #  y1-y2 = a*(x1 - x2) ==>  a = (y1-y2)/(x1-x2)
#     #  b = y1 - a*x1
#
#     a = (sample1['y'] - sample2['y']) / dx
#     b = sample1['y'] - sample1['x'] * a
#     return {'a': a, 'b': b}


# create a fit score between a list of samples and a model (a,b) - with the given cutoff distance
# def scoreModelAgainstSamples(model, samples, cutoff_dist=20):
#     # predict the y using the model and x samples, per sample, and sum the abs of the distances between the real y
#     # with truncation of the error at distance cutoff_dist
#
#     totalScore = 0
#     for sample_i in range(0, len(samples) - 1):
#         sample = samples[sample_i]
#         pred_y = model['a'] * sample['x'] + model['b']
#         score = min(abs(sample['y'] - pred_y), cutoff_dist)
#         totalScore += score
#
#     # print("model ",model, " score ", totalScore)
#     return totalScore


# the function that runs the ransac algorithm (serially)
# gets as input the number of iterations to use and the cutoff_distance for the fit score
# def parallel_ransac(samples, iterations, cutoff_dist):
#     # runs ransac algorithm for the given amount of iterations, where in each iteration it:
#     # 1. randomly creates a model from the samples by calling m = modelFromSamplesFunc(samples)
#     # 2. calculating the score of the model against the sample set
#     # 3. keeps the model with the best score
#     # after all iterations are done - returns the best model and score
#
#     min_m = {}
#     min_score = -1
#     for i in range(1, iterations):
#         if i % 10 == 0:
#             print(i)
#         sample1, sample2 = get_random_sample_pair(samples)
#         m = modelFromSamplePair(sample1, sample2)
#         score = scoreModelAgainstSamples(m, samples, cutoff_dist)
#
#         if min_score < 0 or score < min_score:
#             min_score = score
#             min_m = m
#
#     return {'model': min_m, 'score': min_score}


# ======== some basic pyspark example ======
def some_basic_pyspark_example():
    from pyspark import SparkContext
    num_cores_to_use = 8  # depends on how many cores you have locally. try 2X or 4X the amount of HW threads

    # now we create a spark context in local mode (i.e - not on cluster)
    sc = SparkContext("local[{}]".format(num_cores_to_use), "My First App")


    # function we will use in parallel
    def square_num(x):
        return x * x


    rdd_of_num = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    rdd_of_num_squared = rdd_of_num.map(square_num)

    sum_of_squares = rdd_of_num_squared.reduce(lambda a, b: a + b)

    # if you want to use the DataFrame interface - you need to create a SparkSession around the spark context:
    from pyspark.sql import SparkSession
    session = SparkSession(sc)

    # create dataframe from the rdd of the numbers (call the column my_numbers)
    df = session.createDataFrame(rdd_of_num, ['my_numbers'])
    df = df.withColumn('squares', df['my_numbers'] * df['my_numbers'])
    sum_of_squares = df['squared'].sum()
