import time as time
import pandas as pd

from src.ransac_parallel import extract_data
from src.utils import init_spark, round_up_to_even, init_os_environ, get_case_data, calculate_model_distance


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

    while pairs_to_add != 0:
        samples_to_take = round_up_to_even(pairs_to_add * 2.005)

        random_samples = samples.rdd.takeSample(True, samples_to_take)

        x = 1

        for row_idx_1 in range(samples_to_take // 2):
            row_idx_2 = row_idx_1 + samples_to_take // 2

            if random_samples[row_idx_1]['x'] - random_samples[row_idx_2]['x'] != 0:
                random_sample_pairs_df = random_sample_pairs_df.append({
                    "x1": random_samples[row_idx_1]['x'],
                    "y1": random_samples[row_idx_1]['y'],
                    "x2": random_samples[row_idx_2]['x'],
                    "y2": random_samples[row_idx_2]['y'],
                }, ignore_index=True)

                if num_of_pairs - len(random_sample_pairs_df) == 0:
                    break

        pairs_to_add = num_of_pairs - len(random_sample_pairs_df)

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
    models_df = spark.createDataFrame(models)

    print("\n")
    print(samples.rdd.getNumPartitions())
    print(models_df.rdd.getNumPartitions())

    samples.explain()
    models_df.explain()

    samples.collect()
    models_df.collect()

    samples.explain()
    models_df.explain()

    print("\n")
    print(samples.rdd.getNumPartitions())
    print(models_df.rdd.getNumPartitions())

    totalScore = samples.withColumn('pred_y', models_df['a'] * samples['x'] + models_df['b'])

    x = 1

    return 0


def parallel_ransac(file_path, iterations, cutoff_dist):
    spark, sc = init_spark()

    samples_df = extract_data(spark, file_path)

    random_sample_pairs = get_random_sample_pairs(samples=samples_df, num_of_pairs=iterations)

    models = create_models_from_sample_pairs(sample_pairs=random_sample_pairs)

    result = calc_models_scores_against_samples(spark=spark, samples=samples_df, models=models, cutoff_dist=cutoff_dist)

    #  TODO:  Need This? & Remove
    # samples_df.persist()
    # samples_df.printSchema()
    # samples_df.show(truncate=False)


    return result


def run_ransac(path_to_samples_csv, a, b):
    """--------  Run RANSAC  --------"""

    start = time.time()

    best_model = parallel_ransac(path_to_samples_csv, iterations=5000, cutoff_dist=20)

    end = time.time()

    """--------  Stats  --------"""

    eDistance = calculate_model_distance(original_model={ 'a': a, 'b': b }, best_model=best_model['model'])

    print("\n Run Time {:.3f}\n".format(end - start))

    print("Parallel Algorithm model Stats: - \n")
    print("best model -")
    print(best_model)
    print("The Euclidean distance:  {}".format(eDistance))
    print("********************************")

    # now plot the model
    # samples = read_samples(path_to_samples_csv)
    # plot_model_and_samples(best_model, samples)

    return eDistance


if __name__ == '__main__':
    init_os_environ()
    init_spark()

    CASE_NUM = 2

    path_to_samples_csv, a, b = get_case_data(case_num=CASE_NUM)

    eDistance_original = 0

    eDistance_parallel = run_ransac(path_to_samples_csv=path_to_samples_csv, a=a, b=b)
    # eDistance_parallel = 0

    #  Check who generated better model

    if eDistance_parallel <= eDistance_original:
        print("Parallel RANSAC version generated better model")
    else:
        print("Serial   RANSAC version generated better model")

    while True:
        x = 1
