import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import csv
import time
import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn import model_selection, metrics
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
from xgboost import XGBRegressor

import lightgbm as lgb

from Numerai.feature_neutralization import *
from Numerai.NewData.utils_new import *

PRED_NAME_NEUT_PER_ERA = f'prediction_neutralized_per_era'


# Read the csv file into a pandas Dataframe as float16 to save space
def read_csv(file_path, rows_num=0, skip_rows_num=0, load_val=False):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    # Memory constrained? Try this instead (slower, but more memory efficient)
    # see https://forum.numer.ai/t/saving-memory-with-uint8-features/254

    # dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    # df = pd.read_csv(file_path, dtype=dtypes, index_col=0)

    # dtypes = {f"target": np.float16}
    # to_uint8 = lambda x: np.uint8(float(x) * 4)
    # converters = {x: to_uint8 for x in column_names if x.startswith('feature')}
    # df = pd.read_csv(file_path, dtype=dtypes, converters=converters)

    if load_val:
        dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
        read_rows_skip = pd.read_csv('C:/Users/Ilias/Desktop/Numer.ai/numerai_datasets/validation_indexes.csv')
        skip_rows = list(read_rows_skip['0'])
        # extend skip_rows list to include future extra eras
        skip_rows = skip_rows + list(range((skip_rows[-1] + 2), np.int(10e6)))
        df = pd.read_csv("C:/Users/Ilias/Desktop/Numer.ai/numerai_datasets/numerai_tournament_data.csv",
                         skiprows=skip_rows, dtype=dtypes)
        # df[df['era'] == 'era852']
        # df[df['era'] == 'eraX']
        df.drop(index=56260, inplace=True)
        df.drop(index=137778, inplace=True)
    else:
        if rows_num == 0:
            dtypes = {f"target": np.float16}
            to_uint8 = lambda x: np.uint8(float(x) * 4)
            converters = {x: to_uint8 for x in column_names if x.startswith('feature')}
            df = pd.read_csv(file_path, skiprows=[i for i in range(1, skip_rows_num)], dtype=dtypes,
                             converters=converters)
        else:
            dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}

            df = pd.read_csv(file_path, dtype=dtypes, nrows=rows_num)

    return df


# read the full data
def load_data():
    # Load data
    print("Loading data...")
    # The training data is used to train your model how to predict the targets.
    training_data = pd.read_csv(
        "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz")
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = pd.read_csv(
        "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz")

    feature_names = [
        f for f in training_data.columns if f.startswith("feature")
    ]
    print(f"Loaded {len(feature_names)} features")

    return training_data, tournament_data, feature_names


"""
save and load dtypes object for reading objects(csvs, lists, models etc.)
"""


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# convenience method for scoring
def score(df, pred_name, target_name='target'):
    return spearman(df[pred_name], df[target_name])


def spearman(y_true, y_pred):
    return spearmanr(y_pred, y_true).correlation


def corr_score(df, pred_name, target_name='target', group_name='era'):
    # Check the per-era correlations on the validation set (out of sample)
    correlations = df.groupby(group_name).apply(lambda x: score(x, pred_name, target_name))
    return correlations


def sharpe_score(correlations):
    # Check the "sharpe" ratio on the validation set
    sharpe = correlations.mean() / correlations.std(ddof=0)
    return sharpe


# get indices corresponding to each era
def get_era_idx(df, col='era'):
    era_lst = df[col].unique()
    era_idx = [df[df[col] == x].index for x in era_lst]
    return era_idx


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


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def get_feature_neutral_mean(df, pred_name, target_name):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [pred_name],
                                          feature_cols)[pred_name]
    scores = df.groupby("era").apply(
        lambda x: spearman(x["neutral_sub"], x[target_name])).mean()
    return np.mean(scores)


def richards_dependence(df, target_col, era_col, prediction_col):
    scores_by_era = df.groupby(era_col).apply(
        lambda d: d[[prediction_col, target_col]].corr()[target_col][0]
    )

    # these need to be ranked within era so "error" makes sense
    df[prediction_col] = df.groupby(era_col)[prediction_col].rank(pct=True)
    df[target_col] = df.groupby(era_col)[target_col].rank(pct=True)

    df["era_score"] = df[era_col].map(scores_by_era)

    df["error"] = (df[target_col] - df[prediction_col]) ** 2
    df["1-error"] = 1 - df["error"]

    # Returns the correlation of the 1-error with the era_score
    # i.e. how dependent/correlated each prediction is with its era_score
    return df[["1-error", "era_score"]].corr()


"""
Now that we did all that let's develop a function for modularity
"""


# Swap function
def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


def change_top(df, model1, model2, topx):
    df['rank1'] = df[model1].rank(ascending=False, method='first').astype('int')
    df['rank2'] = df[model2].rank(ascending=False, method='first').astype('int')

    rank_list_2 = df['rank2'].tolist()

    for rank in np.arange(1, topx + 1):
        # find which id corresponds to that position
        id_kept = df.loc[df['rank1'] == rank, 'id'].values[0]

        rank2 = df.loc[df['id'] == id_kept, 'rank2'].values[0]
        pos1 = rank_list_2.index(rank2)

        pos2 = rank_list_2.index(rank)

        new_list = rank_list_2.copy()

        new_list = swapPositions(new_list, pos1, pos2)

        df['new_rank'] = new_list

    # after this all said and done we move on to the ranks to correlations part
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['new_corrs'] = scaler.fit_transform(np.array(df['new_rank']).reshape(-1, 1))

    return df


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
def plot_corrs_per_era(df, pred_name='prediction', target_name='target', group_name='era'):
    val_corrs = corr_score(df, pred_name, target_name, group_name)
    plt.figure()
    ax = sns.barplot(x=val_corrs.index.str.slice(3), y=val_corrs)
    plt.show()


# Feature Neutralization and plot the results
def plot_feature_neutralization(tour_df, neut_percent,
                                target_name='target', pred_name='prediction',
                                group_name='era', full=False, show_metrics=False,
                                scores_on_val2=False, legend_title=None):
    if full == False:
        validation_data = tour_df[tour_df.data_type == "validation"]
    else:
        val2_eras = list(range(197, 213))
        val2_eras = ['era' + str(x) for x in val2_eras]
        validation_data = tour_df[tour_df['era'].isin(val2_eras)]

    # Plot feature exposures
    feat_exps, feats = feature_exposures(validation_data, pred_name)

    plt.figure()
    ax = sns.barplot(x=feats, y=feat_exps)
    ax.legend(title='Max_feature_exposure : {}\n'
                    'Mean feature exposure : {}'.format(max_feature_exposure(validation_data, pred_name),
                                                        feature_exposure(validation_data, pred_name)))
    plt.show()

    val_corrs = corr_score(validation_data, pred_name, target_name, group_name)

    val_sharpe = sharpe_score(val_corrs)

    # Plot the feature exposures with neutralization
    validation_data[pred_name + '_neutralized'] = neutralize_short(validation_data,
                                                                   prediction_name=pred_name,
                                                                   proportion=neut_percent)

    feat_exps, feats = feature_exposures(validation_data, pred_name + '_neutralized')

    plt.figure()
    ax1 = sns.barplot(x=feats, y=feat_exps)
    ax1.legend(title='Max_feature_exposure : {}\n'
                     'Mean feature exposure : {}'.format(
        max_feature_exposure(validation_data, pred_name + '_neutralized'),
        feature_exposure(validation_data, pred_name + '_neutralized')))
    plt.show()

    val_corrs_neut = corr_score(validation_data, pred_name + '_neutralized', target_name, group_name)
    val_sharpe_neut = sharpe_score(val_corrs_neut)

    # plot and print correlations per era
    if show_metrics:
        metrics = print_metrics(tour_df=validation_data, pred_name=pred_name, long_metrics=False)
        if scores_on_val2:
            metrics = print_metrics(tour_df=validation_data, pred_name=pred_name,
                                    long_metrics=False, scores_on_val2=True)

    # Plot the feature exposures with neutralization per era
    validation_data[PRED_NAME_NEUT_PER_ERA] = neutralize(df=validation_data, columns=[pred_name],
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

    val_corrs_neut_per_era = corr_score(validation_data, PRED_NAME_NEUT_PER_ERA, target_name, group_name)
    val_sharpe_neut_per_era = sharpe_score(val_corrs_neut_per_era)

    # plot and print correlations per era
    if show_metrics:
        metrics = print_metrics(tour_df=validation_data, pred_name=pred_name, long_metrics=False)
        if scores_on_val2:
            metrics = print_metrics(tour_df=validation_data, pred_name=pred_name,
                                    long_metrics=False, scores_on_val2=True)

    plt.figure()
    ax1 = sns.barplot(x=val_corrs.index.str.slice(3), y=val_corrs)
    ax1.legend(title=legend_title[0])
    plt.show()

    if show_metrics:
        metrics = print_metrics(tour_df=validation_data, pred_name=pred_name + '_neutralized', long_metrics=False)
        if scores_on_val2:
            metrics = print_metrics(tour_df=validation_data, pred_name=pred_name + '_neutralized',
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


def print_metrics(train_df=None, val_df=None, tour_df=None, feature_names=None, pred_name='prediction',
                  target_name='target', long_metrics=True, scores_on_val2=False, group_name='era'):
    """
    When you print neutralized metrics train_df has to be None cause we don't
    neutralize our targets on train_df
    feature_names : the columns of the features used. Used only when long_metrics=True,
                    otherwise can skip
    scores_on_val2 : Used only for legacy data, to calculate scores on eras 197-212
    """

    if train_df is not None:
        # Check the per-era correlations on the training set (in sample)
        train_correlations = corr_score(train_df, pred_name, target_name, group_name)
        print(
            f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
    else:
        train_correlations = []

    if val_df is not None:
        '''Out of Fold Validation'''
        # Check the per-era correlations on the oof set (out of fold)
        oof_correlations = corr_score(val_df, pred_name, target_name, group_name)
        print(f"On oof the correlation has mean {oof_correlations.mean()} and "
              f"std {oof_correlations.std(ddof=0)}")
    else:
        oof_correlations = []

    if tour_df is not None:
        if scores_on_val2 == False:
            """Validation Metrics"""
            # Check the per-era correlations on the validation set (out of sample)
            validation_data = tour_df[tour_df.data_type == "validation"]
            validation_correlations = corr_score(validation_data, pred_name, target_name, group_name)
            print(f"On validation the correlation has mean {validation_correlations.mean()} and "
                  f"std {validation_correlations.std(ddof=0)}")
        else:
            val2_eras = list(range(197, 213))
            val2_eras = ['era' + str(x) for x in val2_eras]
            validation_data = tour_df[tour_df[group_name].isin(val2_eras)]
            validation_correlations = corr_score(validation_data, pred_name, target_name, group_name)
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
            feature_exposures = val_df[feature_names].apply(lambda d: spearman(val_df[pred_name], d),
                                                            axis=0)
            max_per_era = val_df.groupby(group_name).apply(
                lambda d: d[feature_names].corrwith(d[pred_name]).abs().max())
            max_feature_exposure = max_per_era.mean()
            print(f"Max Feature Exposure for oof: {max_feature_exposure}")

        # Check the feature exposure of your validation predictions
        feature_exposures = validation_data[feature_names].apply(
            lambda d: spearman(validation_data[pred_name], d),
            axis=0)
        max_per_era = validation_data.groupby(group_name).apply(
            lambda d: d[feature_names].corrwith(d[pred_name]).abs().max())
        max_feature_exposure = max_per_era.mean()
        print(f"Max Feature Exposure for validation: {max_feature_exposure}")

        if val_df is not None:
            # Check feature neutral mean for oof
            feature_neutral_mean = get_feature_neutral_mean(val_df, pred_name, target_name)
            print(f"Feature Neutral Mean for oof is {feature_neutral_mean}")

        # Check feature neutral mean for validation
        feature_neutral_mean = get_feature_neutral_mean(validation_data, pred_name, target_name)
        print(f"Feature Neutral Mean for validation is {feature_neutral_mean}")

    return [oof_correlations, validation_correlations, oof_sharpe, validation_sharpe]


# # get models into a  list for iteration on them
# def get_model_lst(num_models=1, folder_name=None):
#     model_lst = []
#     for cv_num in range(num_models):
#         model_lst.append(folder_name + 'model_{}.xgb'.format(cv_num))
#     print(model_lst)
#
#     return model_lst


# FN on either tournament or validation data
def run_feature_neutralization(df=None, predictions_total=None,
                               target_name='target', pred_name='prediction', group_name='era',
                               proportion=0.5, neut_type='short', no_fn=False):
    assert (neut_type in ['short', 'perera'],
            'Wrong keyword given for neut_type. Needed ''short'' or ''perera'' ')
    if no_fn:
        preds = predictions_total
    else:
        # run only for FN
        # choose loading from predictions_csv or from models predictions
        # tournament_data_low_mem[pred_name] = predictions_saved_df['prediction_kazutsugi'] # predictions from csv

        # fill Nans
        df.loc[:, target_name] = df.loc[:, target_name].fillna(0.5)

        df[pred_name] = predictions_total  # predictions from model

        validation_data = df[df['data_type'] == 'validation']
        val_corrs = corr_score(validation_data, pred_name, target_name, group_name)
        sharpe = sharpe_score(val_corrs)
        print('Validation correlations : {}\n'
              'Validation sharpe : {}'.format(val_corrs.mean(), sharpe))
        metrics = print_metrics(tour_df=df, pred_name=pred_name, target_name=target_name,
                                group_name=group_name, long_metrics=False)
        metrics = print_metrics(tour_df=df, pred_name=pred_name, long_metrics=False,
                                group_name=group_name, target_name=target_name, scores_on_val2=True)

        # run only for FN

        # DEFINE FEATURE NEUTRALIZATION PERCENTAGE
        # CAREFUL THIS IS DIFFERENT BETWEEN SUBMISSIONS
        # neut_type has to be either 'short' for neutralize_short()
        #                     either 'perera' for neutralize()
        if neut_type == 'short':
            df[pred_name + '_neutralized'] = neutralize_short(df,
                                                              prediction_name=pred_name,
                                                              proportion=proportion)
        elif neut_type == 'perera':
            df[pred_name + '_neutralized'] = neutralize(df=df, columns=[pred_name],
                                                        extra_neutralizers=df.columns[
                                                            df.columns.str.startswith('feature')],
                                                        proportion=proportion, normalize=True, era_col=group_name)

        validation_data = df[df['data_type'] == 'validation']
        val_corrs = corr_score(validation_data, pred_name + '_neutralized', target_name, group_name)
        sharpe = sharpe_score(val_corrs)
        print('Validation correlations : {}\n'
              'Validation sharpe : {}'.format(val_corrs.mean(), sharpe))

        # metrics will differ somewhat from training notebook cause here we neutralized the whole tournament_data
        # for submission purposes, while in the training notebook we neutralize only the validation_data.
        metrics = print_metrics(tour_df=df, pred_name=pred_name + '_neutralized', target_name=target_name,
                                group_name=group_name, long_metrics=False)
        metrics = print_metrics(tour_df=df, pred_name=pred_name + '_neutralized', long_metrics=False,
                                group_name=group_name, target_name=target_name, scores_on_val2=True)

        # Rescale into [0,1] range keeping rank
        minmaxscaler = MinMaxScaler(feature_range=(0, 0.999999))
        minmaxscaler.fit(np.array(df[pred_name + '_neutralized']).reshape(-1, 1))
        df[pred_name + '_neutralized'] = minmaxscaler.transform(
            np.array(df[pred_name + '_neutralized']).reshape(-1, 1))

        # preds = df[pred_name+'_neutralized'].copy()
        preds = df[pred_name + '_neutralized'].values  # np.array of predictions
    return preds
