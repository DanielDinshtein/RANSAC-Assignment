import time as time

from ransac_excercise import ransac
from ransac_parallel import parallel_ransac

from utils import read_samples, get_case_data, calculate_model_distance


# ========= run ransac ==============


def run_original(path_to_samples_csv, a, b):
    """--------  Run RANSAC  --------"""

    start = time.time()

    samples = read_samples(path_to_samples_csv)

    best_model = ransac(samples, iterations=5000, cutoff_dist=20)

    end = time.time()

    """--------  Stats  --------"""

    eDistance = calculate_model_distance(original_model={ 'a': a, 'b': b }, best_model=best_model['model'])

    print("\n Run Time {:.3f}\n".format(end - start))

    print("Serial Algorithm model Stats :")
    print("best model -")
    print(best_model)
    print("The Euclidean distance:  {}".format(eDistance))
    print("********************************")

    # now plot the model
    # plot_model_and_samples(best_model, samples)

    return eDistance


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


# ============ main =================

if __name__ == '__main__':

    CASE_NUM = 1

    path_to_samples_csv, a, b = get_case_data(case_num=CASE_NUM)

    eDistance_original = run_original(path_to_samples_csv=path_to_samples_csv, a=a, b=b)
    eDistance_parallel = run_ransac(path_to_samples_csv=path_to_samples_csv, a=a, b=b)

    #  Check who generated better model

    if eDistance_parallel <= eDistance_original:
        print("Parallel RANSAC version generated better model")
    else:
        print("Serial   RANSAC version generated better model")

    while True:
        x = 1
