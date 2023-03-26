from basic_functions import get_era_idx, corr_score, sharpe_score
from predictions import predict_in_batch, predict_in_era_batch

import pandas as pd

import seaborn as sns

from xgboost import XGBRegressor
from scipy.stats import spearmanr

PREDICTION_NAME = 'prediction'
TARGET_NAME = 'TARGET'


def era_boost(train_df, tour_df, X, y, _trees_per_step, _iterations, _era_quantile):
    worst_eras_total = []
    era_scores_train = pd.Series(index=train_df['era'].unique())
    era_scores_tour = pd.Series(index=tour_df['era'].unique())
    era_diff_tour = pd.Series(index=tour_df['era'].unique())
    era_diff_tour_total_lst = []
    for i in range(_iterations):
        if i == 0:
            print('Training model ...')
            _model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=_trees_per_step,
                                  n_jobs=-1, colsample_bytree=0.1, tree_method='gpu_hist')  # , tree_method='gpu_hist'
            _model.fit(X, y)
            # , eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True
        else:
            print('Adding trees and training on worst eras ... Iter : {}'.format(i))
            _model.n_estimators += _trees_per_step
            booster = _model.get_booster()
            _model.fit(worst_eras_df[X.columns], worst_eras_df[TARGET_NAME], xgb_model=booster)

        batch_pred = 20000
        preds_train = predict_in_batch(_model, X, batch_pred)
        train_df[PREDICTION_NAME] = preds_train

        preds_tour = predict_in_batch(_model, tour_df[X.columns], batch_pred)
        tour_df[PREDICTION_NAME] = preds_tour

        for era in era_scores_train.index:
            era_df = train_df[train_df['era'] == era]
            era_scores_train[era] = spearmanr(era_df[PREDICTION_NAME], era_df[TARGET_NAME]).correlation

        for era in era_scores_tour.index:
            era_df = tour_df[tour_df['era'] == era]
            era_scores_tour[era] = spearmanr(era_df[PREDICTION_NAME], era_df[TARGET_NAME]).correlation

        if i > 0:
            for era in era_scores_tour.index:
                era_diff_tour[era] = era_scores_tour[era] - era_scores_tour_prior[era]
                dic = {'iteration': 'it' + str(i), 'era': '_' + era[3:], 'correlation_changes': era_diff_tour[era]}
                era_diff_tour_total_lst.append(dic)
            # plt.figure()
            # sns.barplot(x=era_diff_tour.index.str.slice(3), y=era_diff_tour)
            # plt.show()

        # keep the current values to be used as priors in the next iter
        era_scores_tour_prior = era_scores_tour.copy()

        # find worst scores in each era
        era_scores_train.sort_values(inplace=True)
        worst_eras = era_scores_train[era_scores_train <= era_scores_train.quantile(_era_quantile)].index
        worst_eras_df = train_df[train_df['era'].isin(worst_eras)]
        worst_eras_total.append(worst_eras.to_list())

    era_diff_tour_total_df = pd.DataFrame(era_diff_tour_total_lst)

    return worst_eras_total, _model, era_diff_tour_total_df


def era_boost_val_criterion(model, train_df, tour_df, X, y, _trees_per_step, _era_quantile, stop_criterion=20):
    era_idx_train = get_era_idx(train_df)
    era_idx_val = get_era_idx(tour_df)

    worst_eras_total = []
    era_scores_train = pd.Series(index=train_df['era'].unique())
    era_scores_tour = pd.Series(index=tour_df['era'].unique())
    era_diff_tour = pd.Series(index=tour_df['era'].unique())
    era_diff_tour_total_lst = []

    validation_prior = 0
    sharpe_prior = 0

    keep_sharpe_lst = []
    pd.DataFrame(columns=['iteration', 'sharpe_prior', 'sharpe_present'])

    iter_count = 0
    prior_larger = 0
    loop_condition: bool = True

    while loop_condition:
        if iter_count == 0:
            print('Training model ...')
            _model = model
            _model.fit(X, y)
            # , eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True
        else:
            print('Adding trees and training on worst eras ... Iter : {}'.format(iter_count))
            _model.n_estimators += _trees_per_step
            booster = _model.get_booster()
            _model.fit(worst_eras_df[X.columns], worst_eras_df[TARGET_NAME], xgb_model=booster)

        preds_train = predict_in_era_batch(_model, X, era_idx=era_idx_train, rank_per_era=True)
        train_df[PREDICTION_NAME] = preds_train

        preds_tour = predict_in_era_batch(_model, tour_df[X.columns], era_idx=era_idx_val, rank_per_era=True)
        tour_df[PREDICTION_NAME] = preds_tour

        validation_data = tour_df[tour_df['data_type'] == 'validation']
        validation_corrs = corr_score(validation_data, PREDICTION_NAME)
        validation_corrs_mean = validation_corrs.mean()
        sharpe = sharpe_score(validation_corrs)

        update_condition = sharpe_prior < sharpe

        if not update_condition:
            prior_larger += 1

        # loop_condition = prior_larger < 15
        loop_condition = iter_count < stop_criterion

        print('Prior {} and present {}'.format(sharpe_prior, sharpe))

        keep_sharpe = {'iteration': iter_count, 'sharpe_prior': sharpe_prior, 'sharpe_present': sharpe}
        keep_sharpe_lst.append(keep_sharpe)

        validation_prior = validation_corrs_mean
        sharpe_prior = sharpe

        for era in era_scores_train.index:
            era_df = train_df[train_df['era'] == era]
            era_scores_train[era] = spearmanr(era_df[PREDICTION_NAME], era_df[TARGET_NAME]).correlation

        for era in era_scores_tour.index:
            era_df = tour_df[tour_df['era'] == era]
            era_scores_tour[era] = spearmanr(era_df[PREDICTION_NAME], era_df[TARGET_NAME]).correlation

        if iter_count > 0:
            for era in era_scores_tour.index:
                era_diff_tour[era] = era_scores_tour[era] - era_scores_tour_prior[era]
                dic = {'iteration': 'it' + str(iter_count), 'era': '_' + era[3:], 'correlation_changes': era_diff_tour[era]}
                era_diff_tour_total_lst.append(dic)
            # plt.figure()
            # sns.barplot(x=era_diff_tour.index.str.slice(3), y=era_diff_tour)
            # plt.show()

        # keep the current values to be used as priors in the next iter
        era_scores_tour_prior = era_scores_tour.copy()

        # find worst scores in each era
        era_scores_train.sort_values(inplace=True)
        worst_eras = era_scores_train[era_scores_train <= era_scores_train.quantile(_era_quantile)].index
        worst_eras_df = train_df[train_df['era'].isin(worst_eras)]
        worst_eras_total.append(worst_eras.to_list())

        iter_count += 1

    era_diff_tour_total_df = pd.DataFrame(era_diff_tour_total_lst)
    keep_sharpe_df = pd.DataFrame(keep_sharpe_lst)

    keep_sharpe_df_for_lineplot = pd.melt(keep_sharpe_df, ['iteration'])
    sns.lineplot(x='iteration', y='value', hue='variable', data=keep_sharpe_df_for_lineplot)

    return worst_eras_total, _model, era_diff_tour_total_df, keep_sharpe_df
