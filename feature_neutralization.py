import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
import joblib
from tqdm.notebook import tqdm

TARGET_NAME = f'target'
PREDICTION_NAME = f'prediction'
PREDICTION_NAME_NEUTRALIZED = f'prediction_neutralized'


def neutralize_short(df, prediction_name=None, by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith('feature')]

    scores = df[prediction_name].copy()
    # if you dont take .copy() it changes the original df
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))

    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores.values))
    return scores / scores.std()


# to neutralize a column in a df by many other columns on a per-era basis
def neutralize(df,
               columns,
               extra_neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - .5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


@tf.function(experimental_relax_shapes=True, experimental_compile=True)
def exposures(x, y):
    x = x - tf.math.reduce_mean(x, axis=0)
    x = x / tf.norm(x, axis=0)
    y = y - tf.math.reduce_mean(y, axis=0)
    y = y / tf.norm(y, axis=0)
    return tf.matmul(x, y, transpose_a=True)


@tf.function(experimental_relax_shapes=True)
def train_loop_body(model, feats, pred, target_exps):
    with tf.GradientTape() as tape:
        exps = exposures(feats, pred[:, None] - model(feats, training=True))
        loss = tf.reduce_sum(tf.nn.relu(tf.nn.relu(exps) - tf.nn.relu(target_exps)) +
                             tf.nn.relu(tf.nn.relu(-exps) - tf.nn.relu(-target_exps)))
    return loss, tape.gradient(loss, model.trainable_variables)


def train_loop(model, optimizer, feats, pred, target_exps, era):
    for i in range(1000000):
        loss, grads = train_loop_body(model, feats, pred, target_exps)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if loss < 1e-7:
            break
        if i % 10000 == 0:
            tqdm.write(f'era: {era[3:]} loss: {loss:0.7f}', end='\r')


def reduce_exposure(prediction, features, max_exp, era, weights=None):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(310),
        tf.keras.experimental.LinearModel(use_bias=False),
    ])
    feats = tf.convert_to_tensor(features - 0.5, dtype=tf.float32)
    pred = tf.convert_to_tensor(prediction, dtype=tf.float32)
    if weights is None:
        optimizer = tf.keras.optimizers.Adamax()
        start_exp = exposures(feats, pred[:, None])
        target_exps = tf.clip_by_value(start_exp, -max_exp, max_exp)
        train_loop(model, optimizer, feats, pred, target_exps, era)
    else:
        model.set_weights(weights)
    return pred[:, None] - model(feats), model.get_weights()


def reduce_all_exposures(df, column=["prediction"], neutralizers=None,
                         LM_CACHE_FILE=None,
                         normalize=True,
                         gaussianize=True,
                         era_col="era",
                         max_exp=0.1):  ###<-----SELECT YOUR MAXIMUM FEATURE EXPOSURE HERE###
    if neutralizers is None:
        neutralizers = [x for x in df.columns if x.startswith("feature")]
    neutralized = []
    if LM_CACHE_FILE.is_file():
        cache = joblib.load(LM_CACHE_FILE)
        # Remove weights for eraX if we'd accidentally saved it in the past.
        cache.pop("eraX", None)
    else:
        cache = {}
    for era in tqdm(df[era_col].unique()):
        tqdm.write(era, end='\r')
        df_era = df[df[era_col] == era]
        scores = df_era[column].values
        exposure_values = df_era[neutralizers].values

        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                if gaussianize:
                    x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2)[0]

        scores, weights = reduce_exposure(scores, exposure_values,
                                          max_exp, era, cache.get(era))
        if era not in cache and era != "eraX":
            cache[era] = weights
            joblib.dump(cache, LM_CACHE_FILE)
        scores /= tf.math.reduce_std(scores)
        scores -= tf.reduce_min(scores)
        scores /= tf.reduce_max(scores)
        neutralized.append(scores.numpy())

    predictions = pd.DataFrame(np.concatenate(neutralized),
                               columns=column, index=df.index)
    return predictions
