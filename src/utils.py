import os
import math
import json
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType


# ========= utility functions ============


def get_os_variables():
    with open("src/config.json", 'r') as config_file:
        config_data = json.load(config_file)

    os_env_variables = config_data["os_env_variables"]

    java_var = os_env_variables["java"]
    spark_var = os_env_variables["spark"]
    hadoop_var = os_env_variables["hadoop"]

    config_file.close()

    return java_var, spark_var, hadoop_var


def get_case_data(case_num):
    """
    Returns the parameters for the main function based on the case number
    :param case_num: The case number params needed
    :return: samples file path  ; a,b of the original model
    """
    cases = {
        "1": {
            "file_path": 'input files\\samples_for_line_a_48.9684912365_b_44.234.csv',
            "a": 48.9684912365,
            "b": 44.234
        },
        "2": {
            "file_path": 'input files\\samples_for_line_a_27.0976088174_b_12.234 (2).csv',
            "a": 27.0976088174,
            "b": 12.234
        },
    }

    case_data = cases[str(case_num)]

    return case_data['file_path'], case_data['a'], case_data['b']


def calculate_model_distance(original_model, best_model):
    """
    Calculate the Euclidean distance between 2 points -
    models parameter - a,b
    :param original_model: The original line model params
    :param best_model: The RANSAC algorithm best model params
    :return The Distance
    """

    p1 = [original_model['a'], original_model['b']]
    p2 = [best_model['a'], best_model['b']]

    eDistance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    return eDistance


def round_up_to_even(num):
    return math.ceil(num / 2.) * 2


# =========    Serially     ============


def read_samples(filename):
    # Read samples from a csv file and returns them as list of sample dictionaries (each sample is dictionary with 'x' and 'y' keys)

    df = pd.read_csv(filename)
    samples = df[['x', 'y']].to_dict(orient='records')
    return samples


def generate_samples(n_samples=1000, n_outliers=50, b=1, output_path=None):
    # generates new samples - samples will consist of n_samples around some line + n_outliers that are not around the same line
    # gets as parameters:
    # n_samples: the number of inlier samples
    # n_outliers: the number of outlier samples
    # b: the b of the line to use ( the slope - a - will be generated randomly)
    # output_path: optional parameter for also writing out the samples into csv

    from sklearn import linear_model, datasets
    X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                          n_informative=1, noise=10,
                                          coef=True, bias=b)

    print(
        "generated samples around model: a = {} b = {} with {} samples + {} outliers".format(coef.item(0), b, n_samples,
                                                                                             n_outliers))
    if n_outliers > 0:
        # Add outlier data
        np.random.seed(0)
        X[:n_outliers] = 2 * np.random.normal(size=(n_outliers, 1))
        y[:n_outliers] = 10 * np.random.normal(size=n_outliers)

    d = { 'x': X.flatten(), 'y': y.flatten() }
    df = pd.DataFrame(data=d)
    samples = []
    for i in range(0, len(X) - 1):
        samples.append({ 'x': X[i][0], 'y': y[i] })
    ref_model = { 'a': coef.item(0), 'b': b }

    if output_path is not None:
        import os
        file_name = os.path.join(output_path, "samples_for_line_a_{}_b_{}.csv".format(coef.item(0), b))
        df.to_csv(file_name)
    return samples, coef, ref_model


def plot_model_and_samples(model, samples):
    import matplotlib.pyplot as plt
    # plt.rcParams['figure.figsize'] = [20, 10]
    plt.figure()
    xs = [s['x'] for s in samples]
    ys = [s['y'] for s in samples]
    x_min = min(xs)
    x_max = max(xs)
    y_min = model['model']['a'] * x_min + model['model']['b']
    y_max = model['model']['a'] * x_max + model['model']['b']
    plt.plot(xs, ys, '.', [x_min, x_max], [y_min, y_max], '-r')
    plt.grid()
    plt.show()


def calc_without_spark(models, samples, cutoff_dist=20):
    min_m = { }
    min_score = -1
    calc_DF = pd.DataFrame(columns=('pred_y', 'distance', 'score'))

    for idx, model in models.iterrows():
        calc_DF['pred_y'] = model['a'] * samples['x'] + model['b']
        calc_DF['distance'] = samples['y'] - calc_DF['pred_y']

        calc_DF['score'] = [dis if dis <= cutoff_dist else cutoff_dist for dis in
                            calc_DF['distance'].abs().values]

        score, m = calc_DF['score'].sum(), { 'a': model['a'], 'b': model['b'] }

        if min_score < 0 or score < min_score:
            min_score = score
            min_m = m

        calc_DF.drop(calc_DF.index, inplace=True)

    return { 'model': min_m, 'score': min_score }


# =========    Parallel     ============


def init_os_environ():
    java_path, spark_path, hadoop_path = get_os_variables()

    os.environ["JAVA_HOME"] = java_path
    os.environ["SPARK_HOME"] = spark_path
    os.environ["HADOOP_HOME"] = hadoop_path


def init_spark():
    spark = SparkSession.builder \
        .appName("Parallel RANSAC") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("OFF")

    return spark, sc


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
