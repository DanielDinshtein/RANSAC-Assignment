import numpy as np
import pandas as pd

import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.sql import Row

from src.utils import init_spark, round_up_to_even, read_samples, calc_without_spark


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
        StructField('a', DoubleType()), StructField('b', DoubleType())
    ])

    df = spark.createDataFrame([], schema=modelsSchema)

    return df


def get_random_sample_pairs(samples, num_of_pairs):
    """
    Picks pairs of random samples from the DataFrame of samples given.
    * It also makes sure they do not have the same x
    :param samples: The Spark DataFrame of samples
    :param num_of_pairs: Number of pairs needed (number of iterations)
    :return: Pandas DataFrame of Pairs of samples - [x1,y1,x2,y2]
    """

    pairs_to_add = num_of_pairs

    random_sample_pairs_df = pd.DataFrame(columns=('x1', 'y1', 'x2', 'y2'))

    while pairs_to_add > 0:
        samples_to_take = round_up_to_even(pairs_to_add * 1.005)

        random_samples_1 = pd.DataFrame(samples.sample(n=samples_to_take, replace=True).values, columns=('x1', 'y1'))
        random_samples_2 = pd.DataFrame(samples.sample(n=samples_to_take, replace=True).values, columns=('x2', 'y2'))

        df = pd.concat([random_samples_1, random_samples_2], axis=1).reset_index(drop=True)

        random_sample_pairs_df = random_sample_pairs_df.append(df)

        random_sample_pairs_df = random_sample_pairs_df[
            random_sample_pairs_df.x1 - random_sample_pairs_df.x2 != 0].reset_index(drop=True)

        pairs_to_add = num_of_pairs - len(random_sample_pairs_df)

    random_sample_pairs_df = random_sample_pairs_df.iloc[:num_of_pairs]

    return random_sample_pairs_df


def create_models_from_sample_pairs(sample_pairs):
    """
    Generate a line models (a,b) from a pairs of (x,y) samples
    :param sample_pairs: Pandas DataFrame of Pairs of samples - [x1,y1,x2,y2]
    :return: Pandas DataFrame of Models |   model = <a,b>
    """

    models_data = pd.DataFrame()

    # avoid division by zero later
    models_data['dx'] = [x1 - x2 if x1 - x2 != 0 else 0.0001 for x1, x2 in sample_pairs[['x1', 'x2']].values]

    # model = <a,b> where y = ax+b
    # so given x1,y1 and x2,y2 =>
    #  y1 = a*x1 + b
    #  y2 = a*x2 + b
    #  y1-y2 = a*(x1 - x2) ==>  a = (y1-y2)/(x1-x2)
    #  b = y1 - a*x1

    models_data['a'] = (sample_pairs['y1'] - sample_pairs['y2']) / models_data['dx']
    models_data['b'] = sample_pairs['y1'] - sample_pairs['x1'] * models_data['a']

    models_data.drop('dx', axis=1, inplace=True)

    return models_data


def calc_models_scores_against_samples(spark, samples, models, cutoff_dist=20):
    CASE_NUM = 1

    if CASE_NUM == 1:
        return ver1_modelsDF_map_reduce(spark=spark, samples=samples, models=models, cutoff_dist=cutoff_dist)

    models_df = spark.createDataFrame(models)

    models_df.persist()


    # y = models_df.cache().collect()


    def calculate_score(model):
        calc_DF = pd.DataFrame()

        # model = model.asDict()

        calc_DF['pred_y'] = model['a'] * samples['x'] + model['b']
        calc_DF['distance'] = samples['y'] - calc_DF['pred_y']
        calc_DF['score'] = [dis if dis <= cutoff_dist else cutoff_dist for dis in
                            calc_DF['distance'].abs().values]
        return calc_DF['score'].sum(), { 'a': model['a'], 'b': model['b'] }
        # return Row(**{ 'score': calc_DF['score'].sum(), 'model': { 'a': model['a'], 'b': model['b'] } })


    # totalScore = models_df.rdd.map(lambda row: calculate_score(row))
    totalScore = models_df.rdd.map(calculate_score)

    # totalScore_cached = totalScore.cache().collect()
    #
    # df = pd.DataFrame(
    #     [{ 'score': row['score'], 'a': row['model']['a'], 'b': row['model']['b'] } for row in totalScore_cached])

    # res = spark.createDataFrame(df)
    # res = totalScore.toDF().show()

    # res.count()

    # y = 2

    result = totalScore.reduce(lambda model_a, model_b: model_a if model_a[0] <= model_b[0] else model_b)

    models_df.unpersist()
    return { 'model': result[1], 'score': result[0] }


def ver1_modelsDF_map_reduce(spark, samples, models, cutoff_dist=20):
    models_df = spark.createDataFrame(models)

    models_df.persist()


    def calculate_score(model):
        calc_DF = pd.DataFrame()

        calc_DF['pred_y'] = model['a'] * samples['x'] + model['b']
        calc_DF['distance'] = samples['y'] - calc_DF['pred_y']
        calc_DF['score'] = [dis if dis <= cutoff_dist else cutoff_dist for dis in
                            calc_DF['distance'].abs().values]
        return calc_DF['score'].sum(), { 'a': model['a'], 'b': model['b'] }


    totalScore = models_df.rdd.map(calculate_score)

    result = totalScore.reduce(lambda model_a, model_b: model_a if model_a[0] <= model_b[0] else model_b)

    models_df.unpersist()

    return { 'model': result[1], 'score': result[0] }


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
    # samples_df = extract_data(spark, file_path)

    samples_df = pd.DataFrame(read_samples(file_path))

    random_sample_pairs = get_random_sample_pairs(samples=samples_df, num_of_pairs=iterations)

    models = create_models_from_sample_pairs(sample_pairs=random_sample_pairs)

    with_spark = True

    if with_spark:
        spark, sc = init_spark()
        result = calc_models_scores_against_samples(spark=spark, samples=samples_df, models=models,
                                                    cutoff_dist=cutoff_dist)

        # sc.stop()
    else:
        result = calc_without_spark(models=models, samples=samples_df, cutoff_dist=cutoff_dist)

    #  TODO:  Need This? & Remove
    # samples_df.persist()
    # samples_df.printSchema()
    # samples_df.show(truncate=False)


    return result
