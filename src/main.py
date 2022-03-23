from ransac_parallel import parallel_ransac


# ========= run ransac ==============

def run_ransac():
    path_to_samples_csv = "C:/Users/Daniel/Desktop/משרות/משימות מראיונות/Mobileye/RANSAC-Assignment/input files/samples_for_line_a_48.9684912365_b_44.234.csv"
    best_model = parallel_ransac(path_to_samples_csv, iterations=5000, cutoff_dist=20)

    # samples = read_samples(path_to_samples_csv)
    # best_model = parallel_ransac(samples, iterations=5000, cutoff_dist=20)
    #
    # # now plot the model
    # plot_model_and_samples(best_model, samples)


# ============ main =================

if __name__ == '__main__':
    run_ransac()
