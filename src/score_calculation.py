import pandas as pd

import pyspark.sql.functions as F


def modelsDF_map_reduce(spark, samples, models, cutoff_dist=20):
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


def samplesDF_map_reduce(spark, samples, models, cutoff_dist=20):
    samples_df = spark.createDataFrame(samples)
    samples_df.persist()

    models['idx'] = [str(i) for i in range(1, 1 + len(models))]
    models_df = spark.createDataFrame(models[['idx', 'a', 'b']])

    samples_df.printSchema()
    samples_df.show(truncate=False)

    x = 1

    models_df = spark.createDataFrame(models)
    models_df.persist()

    models_df = models_df.select(
        F.when(F.abs(samples['y'] - (F.col('a') * samples['x'] + F.col('b'))) > F.lit(cutoff_dist), F.lit(cutoff_dist))) \
        .otherwise()


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


def samplesDF_modelsDF(spark, samples, models, cutoff_dist=20):
    samples_df = spark.createDataFrame(samples)
    samples_df.persist()

    models['idx'] = [str(i) for i in range(1, 1 + len(models))]
    models_df = spark.createDataFrame(models[['idx', 'a', 'b']])
    models_df.persist()


    def merge_both_df(model):
        pred_y = model['a'] * samples['x'] + model['b']
        distance = abs(samples['y'] - pred_y)
        score = [dis if dis <= cutoff_dist else cutoff_dist for dis in distance]

        totalScore = sum(score)

        return model['idx'], model['a'], model['b'], totalScore


    merged_rdd = models_df.rdd.map(lambda model: merge_both_df(model=model))

    models_summary = merged_rdd.toDF(['idx', 'a', 'b', 'totalScore'])

    models_summary_ordered = models_summary.orderBy('totalScore')

    models_summary_ordered.persist().head(1)

    row1 = models_summary_ordered.collect()[0]
    # row1 = models_summary.agg({ 'totalScore': 'min' }).collect()[0]

    models_summary.printSchema()
    models_summary.show(truncate=False)

    samples_df.printSchema()
    samples_df.show(truncate=False)

    x = 1

    models_df = models_df.select(
        F.when(F.abs(samples['y'] - (F.col('a') * samples['x'] + F.col('b'))) > F.lit(cutoff_dist), F.lit(cutoff_dist))) \
        .otherwise()


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


def calc_models_scores_against_samples(spark, samples, models, cutoff_dist=20):
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

    # df = pd.DataFrame(
    #     [{ 'score': row['score'], 'a': row['model']['a'], 'b': row['model']['b'] } for row in totalScore_cached])

    # res = spark.createDataFrame(df)
    # res = totalScore.toDF().show()

    # res.count()
    # y = 2

    result = totalScore.reduce(lambda model_a, model_b: model_a if model_a[0] <= model_b[0] else model_b)

    models_df.unpersist()
    return { 'model': result[1], 'score': result[0] }


def samplesDF_calc(model, samples, cutoff_dist=20):
    # predict the y using the model and x samples, per sample, and sum the abs of the distances between the real y
    # with truncation of the error at distance cutoff_dist

    totalScore = samples.withColumn('pred_y', model['a'] * samples['x'] + model['b']) \
        .withColumn('distance', F.abs(samples['y'] - F.col('pred_y'))) \
        .withColumn('score', F
                    .when(F.col('distance') <= F.lit(cutoff_dist), F.col('distance'))
                    .otherwise(F.lit(cutoff_dist)))

    totalScore = totalScore.select('score').toPandas()['score'].sum()

    return totalScore
