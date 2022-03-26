import os
import math
import numpy as np
import pandas as pd

import json

from pyspark.sql import SparkSession


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
    init_os_environ()

    num_cores_to_use = "4"  # depends on how many cores you have locally. try 2X or 4X the amount of HW threads

    spark = SparkSession.builder \
        .appName("Parallel RANSAC") \
        .config("spark.executor.cores", num_cores_to_use) \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("OFF")

    return spark, sc
