import time as time

from ransac_parallel import parallel_ransac

from ransac_excercise import ransac, read_samples
from utils import plot_model_and_samples


# ========= run ransac ==============


def run_original():
    start = time.time()

    path_to_samples_csv = 'input files/samples_for_line_a_48.9684912365_b_44.234.csv'
    samples = read_samples(path_to_samples_csv)

    best_model = ransac(samples, iterations=5000, cutoff_dist=20)

    end = time.time()
    print("\n* Run Time {:.3f}\n".format(end - start))

    print("Serial Algorithm model Stats: - \n")
    print(best_model)
    print("********************************")

    # now plot the model
    # plot_model_and_samples(best_model, samples)


def run_ransac():
    start = time.time()

    path_to_samples_csv = "C:/Users/Daniel/Desktop/משרות/משימות מראיונות/Mobileye/RANSAC-Assignment/input files/samples_for_line_a_48.9684912365_b_44.234.csv"
    best_model = parallel_ransac(path_to_samples_csv, iterations=5000, cutoff_dist=20)

    end = time.time()
    print("\n* Run Time {:.3f}\n".format(end - start))

    print("Parallel Algorithm model Stats: - \n")
    print(best_model)
    print("********************************")

    # now plot the model
    samples = read_samples(path_to_samples_csv)
    plot_model_and_samples(best_model, samples)


# ============ main =================

if __name__ == '__main__':
    run_original()
    run_ransac()

    print("a = 48.9684912365 \nb = 44.234")
