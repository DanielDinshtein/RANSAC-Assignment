import pandas as pd

from src.utils import read_samples, calc_without_spark
from src.score_calculation import modelsDF_map_reduce, model_samples_rdd_df


# =========    Parallel     ============


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
        samples_to_take = pairs_to_add

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

    models_data = models_data.drop('dx', axis=1).reset_index(drop=True)

    return models_data


def calc_models_scores_against_samples(samples, models, cutoff_dist=20, CALC_CASE=2):
    if CALC_CASE == 1:
        return modelsDF_map_reduce(samples=samples, models=models, cutoff_dist=cutoff_dist)

    if CALC_CASE == 2:
        return model_samples_rdd_df(samples=samples, models=models, cutoff_dist=cutoff_dist)


def parallel_ransac(file_path, iterations, cutoff_dist, calc_case=2):
    # samples_df = extract_data(spark, file_path)

    samples_df = pd.DataFrame(read_samples(file_path))

    random_sample_pairs = get_random_sample_pairs(samples=samples_df, num_of_pairs=iterations)

    models = create_models_from_sample_pairs(sample_pairs=random_sample_pairs)

    with_spark = True

    if with_spark:
        result = calc_models_scores_against_samples(samples=samples_df, models=models,
                                                    cutoff_dist=cutoff_dist, CALC_CASE=calc_case)

    else:
        result = calc_without_spark(models=models, samples=samples_df, cutoff_dist=cutoff_dist)

    return result
