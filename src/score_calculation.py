import pandas as pd

from src.utils import init_spark


def modelsDF_map_reduce(samples, models, cutoff_dist=20):
    spark, sc = init_spark()
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


def model_samples_rdd_df(samples, models, cutoff_dist=20):
    spark, sc = init_spark()

    models['idx'] = [str(i) for i in range(0, len(models))]

    models_rdd = sc.parallelize([[model, samples] for model in models.values.tolist()])


    def calculate_model_score(data):
        model = data[0]
        samples = data[1]

        pred_y = model[0] * samples['x'] + model[1]
        distance = abs(samples['y'] - pred_y)
        score = [dis if dis <= cutoff_dist else cutoff_dist for dis in distance]

        totalScore = sum(score)

        return model[2], model[0], model[1], totalScore


    calculated_rdd = models_rdd.map(lambda model: calculate_model_score(data=model))

    models_summary = calculated_rdd.toDF(['idx', 'a', 'b', 'totalScore'])

    models_summary_ordered = models_summary.orderBy('totalScore')

    result = models_summary_ordered.head(1)[0]

    return { 'model': { 'a': result[1], 'b': result[2] }, 'score': result[3] }
