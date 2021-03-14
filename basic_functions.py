import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import csv
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn import model_selection, metrics
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

TARGET_NAME = f'target'
PREDICTION_NAME = f'prediction'
PREDICTION_NAME_NEUTRALIZED = f'prediction_neutralized'
PRED_NAME_NEUT_PER_ERA = f'prediction_neutralized_per_era'


# convenience method for scoring
def score(df, pred_name):
    return spearman(df[pred_name], df[TARGET_NAME])


def spearman(y_true, y_pred):
    return spearmanr(y_pred, y_true).correlation


def corr_score(df, pred_name):
    # Check the per-era correlations on the validation set (out of sample)
    correlations = df.groupby("era").apply(lambda x: score(x, pred_name))
    return correlations


def sharpe_score(correlations):
    # Check the "sharpe" ratio on the validation set
    sharpe = correlations.mean() / correlations.std(ddof=0)
    return sharpe


# predict in batches to avoid memory issues
# df should contain only the prediction features
# i.e. training_data[feature_names]
def predict_in_batch(model, df, batch):
    predictions = []
    for i in range(0, len(df), batch):
        preds = model.predict(df[i: i + batch])
        predictions.extend(preds)
    return predictions


# predict in era batches to avoid memory issues
# df should contain only the prediction features
# i.e. training_data[feature_names]
def predict_in_era_batch(model, df, era_idx):
    predictions = []
    for era in era_idx:
        preds = model.predict(df.loc[era])
        predictions.extend(preds)
    return predictions


def feature_exposures(df, pred_name):
    feature_names = [f for f in df.columns
                     if f.startswith("feature")]
    exposures = []
    for f in feature_names:
        fe = spearmanr(df[pred_name], df[f])[0]
        exposures.append(fe)
    return [np.array(exposures), feature_names]


def max_feature_exposure(df, pred_name):
    return np.max(np.abs(feature_exposures(df, pred_name)[0]))


def feature_exposure(df, pred_name):
    return np.sqrt(np.mean(np.square(feature_exposures(df, pred_name)[0])))


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


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def get_feature_neutral_mean(df):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME],
                                          feature_cols)[PREDICTION_NAME]
    scores = df.groupby("era").apply(
        lambda x: spearman(x["neutral_sub"], x[TARGET_NAME])).mean()
    return np.mean(scores)


# plot the feature exposures
def plot_feature_exposures(df, pred_name):
    fe = feature_exposures(df, pred_name)
    plt.figure()
    xaxis = [fe[1][x] for x in range(0, len(fe[1]), 2)]
    # print(xaxis)
    # plt.xticks(rotation=90, fontsize=10, ticks=np.arange(0, len(fe[1]), 10))
    ax = sns.barplot(x=fe[1], y=fe[0])
    ax.legend(title='Max_feature_exposure : {}\n'
                    'Mean feature exposure : {}'.format(max_feature_exposure(df, pred_name),
                                                        feature_exposure(df, pred_name)))
    plt.xticks(rotation=90, fontsize=8, ticks=np.arange(0, len(fe[1]), 2), labels=xaxis)
    plt.show()
    return fe


# plot the correlations per era
def plot_corrs_per_era(df, pred_name):
    val_corrs = corr_score(df, pred_name)
    plt.figure()
    ax = sns.barplot(x=val_corrs.index.str.slice(3), y=val_corrs)
    plt.show()


# Feature Neutralization and plot the results
def plot_feature_neutralization(tour_df, neut_percent, full=False, show_metrics=False, scores_on_val2=False,
                                legend_title=None):
    if full == False:
        validation_data = tour_df[tour_df.data_type == "validation"]
    else:
        val2_eras = list(range(197, 213))
        val2_eras = ['era' + str(x) for x in val2_eras]
        validation_data = tour_df[tour_df['era'].isin(val2_eras)]

    # Plot feature exposures
    feat_exps, feats = feature_exposures(validation_data, PREDICTION_NAME)

    plt.figure()
    ax = sns.barplot(x=feats, y=feat_exps)
    ax.legend(title='Max_feature_exposure : {}\n'
                    'Mean feature exposure : {}'.format(max_feature_exposure(validation_data, PREDICTION_NAME),
                                                        feature_exposure(validation_data, PREDICTION_NAME)))
    plt.show()

    val_corrs = corr_score(validation_data, PREDICTION_NAME)

    val_sharpe = sharpe_score(val_corrs)

    # Plot the feature exposures with neutralization
    validation_data[PREDICTION_NAME_NEUTRALIZED] = neutralize_short(validation_data,
                                                                    prediction_name=PREDICTION_NAME,
                                                                    proportion=neut_percent)

    feat_exps, feats = feature_exposures(validation_data, PREDICTION_NAME_NEUTRALIZED)

    plt.figure()
    ax1 = sns.barplot(x=feats, y=feat_exps)
    ax1.legend(title='Max_feature_exposure : {}\n'
                     'Mean feature exposure : {}'.format(
        max_feature_exposure(validation_data, PREDICTION_NAME_NEUTRALIZED),
        feature_exposure(validation_data, PREDICTION_NAME_NEUTRALIZED)))
    plt.show()

    val_corrs_neut = corr_score(validation_data, PREDICTION_NAME_NEUTRALIZED)
    val_sharpe_neut = sharpe_score(val_corrs_neut)

    # plot and print correlations per era
    if show_metrics:
        metrics = print_metrics(tour_df=validation_data, pred_name=PREDICTION_NAME, long_metrics=False)
        if scores_on_val2:
            metrics = print_metrics(tour_df=validation_data, pred_name=PREDICTION_NAME,
                                    long_metrics=False, scores_on_val2=True)

    # Plot the feature exposures with neutralization per era
    validation_data[PRED_NAME_NEUT_PER_ERA] = neutralize(df=validation_data, columns=[PREDICTION_NAME],
                                                         extra_neutralizers=feature_names,
                                                         proportion=neut_percent, normalize=True, era_col='era')

    # validation_data[PRED_NAME_NEUT_PER_ERA] = minmax_scale_values(df=validation_data,
    #                                                               pred_name=PRED_NAME_NEUT_PER_ERA)

    feat_exps, feats = feature_exposures(validation_data, PRED_NAME_NEUT_PER_ERA)

    plt.figure()
    ax1 = sns.barplot(x=feats, y=feat_exps)
    ax1.legend(title='Max_feature_exposure : {}\n'
                     'Mean feature exposure : {}'.format(
        max_feature_exposure(validation_data, PRED_NAME_NEUT_PER_ERA),
        feature_exposure(validation_data, PRED_NAME_NEUT_PER_ERA)))
    plt.show()

    val_corrs_neut_per_era = corr_score(validation_data, PRED_NAME_NEUT_PER_ERA)
    val_sharpe_neut_per_era = sharpe_score(val_corrs_neut_per_era)

    # plot and print correlations per era
    if show_metrics:
        metrics = print_metrics(tour_df=validation_data, pred_name=PREDICTION_NAME, long_metrics=False)
        if scores_on_val2:
            metrics = print_metrics(tour_df=validation_data, pred_name=PREDICTION_NAME,
                                    long_metrics=False, scores_on_val2=True)

    plt.figure()
    ax1 = sns.barplot(x=val_corrs.index.str.slice(3), y=val_corrs)
    ax1.legend(title=legend_title[0])
    plt.show()

    if show_metrics:
        metrics = print_metrics(tour_df=validation_data, pred_name=PREDICTION_NAME_NEUTRALIZED, long_metrics=False)
        if scores_on_val2:
            metrics = print_metrics(tour_df=validation_data, pred_name=PREDICTION_NAME_NEUTRALIZED,
                                    long_metrics=False, scores_on_val2=True)

    plt.figure()
    ax1 = sns.barplot(x=val_corrs_neut.index.str.slice(3), y=val_corrs_neut)
    ax1.legend(title=legend_title[1])
    plt.show()

    if show_metrics:
        metrics = print_metrics(tour_df=validation_data, pred_name=PRED_NAME_NEUT_PER_ERA, long_metrics=False)
        if scores_on_val2:
            metrics = print_metrics(tour_df=validation_data, pred_name=PRED_NAME_NEUT_PER_ERA,
                                    long_metrics=False, scores_on_val2=True)

    plt.figure()
    ax1 = sns.barplot(x=val_corrs_neut_per_era.index.str.slice(3), y=val_corrs_neut_per_era)
    ax1.legend(title=legend_title[2])
    plt.show()


def print_metrics(train_df=None, val_df=None, tour_df=None, feature_names=None, pred_name=None, long_metrics=True,
                  scores_on_val2=False):
    # when you print neutralized metrics train_df has to be None cause we don't
    # neutralize our targets on train_df

    if train_df is not None:
        # Check the per-era correlations on the training set (in sample)
        train_correlations = corr_score(train_df, pred_name)
        print(
            f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
    else:
        train_correlations = []

    if val_df is not None:
        '''Out of Fold Validation'''
        # Check the per-era correlations on the oof set (out of fold)
        oof_correlations = corr_score(val_df, pred_name)
        print(f"On oof the correlation has mean {oof_correlations.mean()} and "
              f"std {oof_correlations.std(ddof=0)}")
    else:
        oof_correlations = []

    if tour_df is not None:
        if scores_on_val2 == False:
            """Validation Metrics"""
            # Check the per-era correlations on the validation set (out of sample)
            validation_data = tour_df[tour_df.data_type == "validation"]
            validation_correlations = corr_score(validation_data, pred_name)
            print(f"On validation the correlation has mean {validation_correlations.mean()} and "
                  f"std {validation_correlations.std(ddof=0)}")
        else:
            val2_eras = list(range(197, 213))
            val2_eras = ['era' + str(x) for x in val2_eras]
            validation_data = tour_df[tour_df['era'].isin(val2_eras)]
            validation_correlations = corr_score(validation_data, pred_name)
            print(f"On validation_2 the correlation has mean {validation_correlations.mean()} and "
                  f"std {validation_correlations.std(ddof=0)}")

    else:
        validation_correlations = []

    if val_df is not None:
        # Check the "sharpe" ratio on the oof set
        oof_sharpe = oof_correlations.mean() / oof_correlations.std(ddof=0)
        print(f"Oof Sharpe: {oof_sharpe}")
    else:
        oof_sharpe = []

    if tour_df is not None:
        # Check the "sharpe" ratio on the validation set
        validation_sharpe = sharpe_score(validation_correlations)
        print(f"Validation Sharpe: {validation_sharpe}")
    else:
        validation_sharpe = []

    if long_metrics == True:
        if val_df is not None:
            # Checking the max drowdown of the oof
            rolling_max = (oof_correlations + 1).cumprod().rolling(window=100,
                                                                   min_periods=1).max()
            daily_value = (oof_correlations + 1).cumprod()
            max_drawdown = -(rolling_max - daily_value).max()
            print(f"max drawdown oof: {max_drawdown}")

        # checking the max drowdown of the validation
        rolling_max = (validation_correlations + 1).cumprod().rolling(window=100,
                                                                      min_periods=1).max()
        daily_value = (validation_correlations + 1).cumprod()
        max_drawdown = -(rolling_max - daily_value).max()
        print(f"max drawdown validation: {max_drawdown}")

        if val_df is not None:
            # Check the feature exposure of your oof predictions
            feature_exposures = val_df[feature_names].apply(lambda d: spearman(val_df[PREDICTION_NAME], d),
                                                            axis=0)
            max_per_era = val_df.groupby("era").apply(
                lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
            max_feature_exposure = max_per_era.mean()
            print(f"Max Feature Exposure for oof: {max_feature_exposure}")

        # Check the feature exposure of your validation predictions
        feature_exposures = validation_data[feature_names].apply(
            lambda d: spearman(validation_data[PREDICTION_NAME], d),
            axis=0)
        max_per_era = validation_data.groupby("era").apply(
            lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
        max_feature_exposure = max_per_era.mean()
        print(f"Max Feature Exposure for validation: {max_feature_exposure}")

    return [oof_correlations, validation_correlations, oof_sharpe, validation_sharpe]


# get models into a list for iteration on them
def get_model_lst(num_models=1, folder_name=None):
    model_lst = []
    for cv_num in range(num_models):
        model_lst.append(folder_name + 'model_{}.xgb'.format(cv_num))
    print(model_lst)

    return model_lst


# predict in batches. XGBRegressor supported only atm
def get_predictions(df=None, num_models=1, folder_name=None):
    model_lst = get_model_lst(num_models=num_models, folder_name=folder_name)
    predictions_total = []
    for cv_num in range(num_models):
        model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=1000, n_jobs=-1, colsample_bytree=0.1,
                             tree_method='gpu_hist', verbosity=0)  # tree_method='gpu_hist',
        model.load_model(model_lst[cv_num])

        # select the feature columns from the tournament data
        X_test = df

        # predict in batches to avoid memory issues
        batch_pred = 20000
        predictions = []
        for i in range(0, len(X_test), batch_pred):
            preds = model.predict(X_test[i: i + batch_pred])
            predictions.extend(preds)

        predictions_total.append(predictions)

    predictions_total = np.mean(predictions_total, axis=0)

    return predictions_total


# FN on either tournament or validation data
def run_feature_neutralization(df=None, predictions_total=None, proportion=0.5, neut_type='short', no_fn=False):
    if no_fn:
        preds = predictions_total
    else:
        # run only for FN
        # choose loading from predictions_csv or from models predictions
        # tournament_data_low_mem[PREDICTION_NAME] = predictions_saved_df['prediction_kazutsugi'] # predictions from csv
        df[PREDICTION_NAME] = predictions_total  # predictions from model

        validation_data = df[df['data_type'] == 'validation']
        val_corrs = corr_score(validation_data, PREDICTION_NAME)
        sharpe = sharpe_score(val_corrs)
        print('Validation correlations : {}\n'
              'Validation sharpe : {}'.format(val_corrs.mean(), sharpe))
        metrics = print_metrics(tour_df=df, pred_name=PREDICTION_NAME, long_metrics=False)
        metrics = print_metrics(tour_df=df, pred_name=PREDICTION_NAME, long_metrics=False, scores_on_val2=True)

        # run only for FN

        # DEFINE FEATURE NEUTRALIZATION PERCENTAGE
        # CAREFUL THIS IS DIFFERENT BETWEEN SUBMISSIONS
        # neut_type has to be either 'short' for neutralize_short()
        #                     either 'perera' for neutralize()
        if neut_type == 'short':
            df[PREDICTION_NAME_NEUTRALIZED] = neutralize_short(df,
                                                               prediction_name=PREDICTION_NAME,
                                                               proportion=proportion)
        elif neut_type == 'perera':
            df[PREDICTION_NAME_NEUTRALIZED] = neutralize(df=df, columns=[PREDICTION_NAME],
                                                         extra_neutralizers=df.columns[
                                                             df.columns.str.startswith('feature')],
                                                         proportion=proportion, normalize=True, era_col='era')
        else:
            print('Error keyword given for neut_type. Needed ''short'' or ''perera'' ')

        validation_data = df[df['data_type'] == 'validation']
        val_corrs = corr_score(validation_data, PREDICTION_NAME_NEUTRALIZED)
        sharpe = sharpe_score(val_corrs)
        print('Validation correlations : {}\n'
              'Validation sharpe : {}'.format(val_corrs.mean(), sharpe))

        # metrics will differ somewhat from training notebook cause here we neutralized the whole tournament_data
        # for submission purposes, while in the training notebook we neutralize only the validation_data.
        metrics = print_metrics(tour_df=validation_data, pred_name=PREDICTION_NAME_NEUTRALIZED, long_metrics=False)
        metrics = print_metrics(tour_df=validation_data, pred_name=PREDICTION_NAME_NEUTRALIZED, long_metrics=False,
                                scores_on_val2=True)

        # Rescale into [0,1] range keeping rank
        minmaxscaler = MinMaxScaler(feature_range=(0, 0.999999))
        minmaxscaler.fit(np.array(df[PREDICTION_NAME_NEUTRALIZED]).reshape(-1, 1))
        df[PREDICTION_NAME_NEUTRALIZED] = minmaxscaler.transform(
            np.array(df[PREDICTION_NAME_NEUTRALIZED]).reshape(-1, 1))

        preds = df[PREDICTION_NAME_NEUTRALIZED].copy()

    return preds
