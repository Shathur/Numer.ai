import numpy as np

from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples

from sklearn.metrics import log_loss, accuracy_score

from Numerai import utils
from Numerai import models
from Numerai.predictions import predict_in_era_batch

from Numerai.NewData.utils_new import *


class Random_Splits(_BaseKFold):
    def __init__(self, n_splits=None):
        self.n_splits = n_splits
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        group_lst = np.unique(groups)
        n_groups = len(group_lst)

        indices = np.arange(n_samples)

        cutoff_eras = n_groups // self.n_splits
        np.random.shuffle(group_lst)

        for i in range(self.n_splits):
            yield (indices[groups.isin(group_lst[i * cutoff_eras: i * cutoff_eras + cutoff_eras])],
                   indices[groups.isin(group_lst[i * cutoff_eras: i * cutoff_eras + cutoff_eras])])


class TimeSeriesSplitGroups(_BaseKFold):
    """
    Code kindly provided by Michael Oliver in the Numer.ai forum
    https://forum.numer.ai/t/era-wise-time-series-cross-validation/791
    """

    def __init__(self, n_splits=None):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_groups))
        indices = np.arange(n_samples)
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:
            yield (indices[groups.isin(group_list[:test_start])],
                   indices[groups.isin(group_list[test_start:test_start + test_size])])


class TimeSeriesSplitGroupsPurged(_BaseKFold):
    """
    Code kindly provided by Michael Oliver in the Numer.ai forum
    https://forum.numer.ai/t/era-wise-time-series-cross-validation/791
    """

    def __init__(self, n_splits=None, embg_grp_num=None):
        if n_splits>1:
            super().__init__(n_splits,shuffle=False, random_state=None)
        else:
            self.n_splits = n_splits
        self.embg_grp_num = embg_grp_num

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        embg_grp_num = self.embg_grp_num
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds : {0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_groups))
        indices = np.arange(n_samples)
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        if n_splits>1:
            for test_start in test_starts:
                yield (indices[groups.isin(group_list[:test_start-embg_grp_num])],
                       indices[groups.isin(group_list[test_start+embg_grp_num : test_start+embg_grp_num + test_size])])
        else:
            yield (indices[groups.isin(group_list[:int(0.8*len(group_list))-embg_grp_num])],
                    indices[groups.isin(group_list[int(0.8*len(group_list))-embg_grp_num:])])


class PurgedKfold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in

    Code was adapted from Advances in Financial ML by Marcos Lopez De Prado.
    Snippet 7.3

    t1 : must be a pd.Series
    X and t1 must have the same index values
    """

    def __init__(self, n_splits=None, t1=None, pctEmbargo=None):
        super(PurgedKfold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0]*self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]

        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i: j]
            maxt1idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxt1idx < X.shape[0]:
                train_indices = np.concatenate((train_indices, indices[maxt1idx + mbrg:]))
            yield train_indices, test_indices


class Windowed_Groups(_BaseKFold):
    def __init__(self, n_splits=None):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None, window_length=4):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        group_lst = np.unique(groups)
        n_groups = len(group_lst)

        indices = np.arange(n_samples)

        eras = range(n_groups - window_length)
        eras = list(eras)
        for i in eras[:]:
            yield (indices[groups == group_lst[i]],
                   indices[groups == group_lst[i + window_length]])


def cvscore(clf, X, y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=None):
    """
    Alternative to cross_val_score for the PurgedKFold class
    cross_val_score will give different results because it passes weights to the fit method, but
    not to the log_loss method.

    scoring : Must belong in ['neg_log_loss', 'accuracy']
    """
    if cvGen is None:
        cvGen = PurgedKfold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    score = []

    for train, test in cvGen.split(X=X):
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)

        score.append(score_)

    return np.array(score)


def cross_validate_train(feature_names, cv_split_data, target_name=TARGET_NAME, train_df=None,
                         tour_df=None, type_of_model='xgb', model_params=None, fit_params=None, save_to_drive=False,
                         save_folder=None, legacy_save=True, calculate_metrics=True, plot_metrics=False):
    """

    :param feature_names: list with feature names
    :param cv_split_data: list of one of the splitters above
                    e.g. time_group_splitter = TimeSeriesSplitGroups(n_splits=4).split(new_training_data, groups=erano_values)
                    cv_split_data = list(time_group_splitter)
    :param target_name : target to train on, defaults to TARGET_NAME global
    :param train_df: train dataset that contains training and oos data
    :param tour_df: validation dataset for metrics visualization
    :param type_of_model: the model to be created and used for training - must be 'xgb' or 'lgb'
    :param model_params: parameters of the model. If None get defaults
    :param fit_params: parameters for the .fit() function. If None get defaults
    :param save_to_drive: True - Save to drive False - Temporarily save
    :param save_folder: Folder to redirect our models
    :param calculate_metrics: return some basic performance metrics like correlations and sharpe.
                    If True returns those metrics, if False returns nothing and just saves the models
    :param plot_metrics: simple sns.barplot of our results
    :return: val_correlations, tour_correlations, val_sharpe_cv, tour_sharpe_cv

    """
    val_corrs_mean_cv = []
    val_corrs_std_cv = []
    tour_corrs_mean_cv = []
    tour_corrs_std_cv = []
    val_sharpe_cv = []
    tour_sharpe_cv = []

    predictions_train_total = []
    predictions_val_total = []
    predictions_tour_total = []

    for cv_count, idx_cv in enumerate(cv_split_data):

        train_data = train_df.iloc[idx_cv[0]]
        val_data = train_df.iloc[idx_cv[1]]

        # if you want to train on the whole data without keeping any for validation
        # remember when you use the whole data to not use validation since we haven't kept any data for it
        # also remember to save to the appropriate folder
        # train_data = pd.concat([train_data, val_data])

        X_train = train_data[feature_names]
        y_train = train_data[target_name]
        X_val = val_data[feature_names]
        y_val = val_data[target_name]

        era_lst_train = train_data['era'].unique()
        era_lst_validation = val_data['era'].unique()

        era_idx_train = [train_data[train_data['era'] == x].index for x in era_lst_train]
        train_data.loc[:, target_name] = train_data.loc[:, target_name].fillna(2)

        era_idx_val = [val_data[val_data['era'] == x].index for x in era_lst_validation]
        val_data.loc[:, target_name] = val_data.loc[:, target_name].fillna(2)

        if ((tour_df is not None) and (not tour_df.empty)):
            era_lst_tour = tour_df['era'].unique()
            era_idx_tour = [tour_df[tour_df['era'] == x].index for x in era_lst_tour]
            # here is also the spot to check the target for Nan and fill them with 0.5
            tour_df.loc[:, target_name] = tour_df.loc[:, target_name].fillna(0.5)

        print('********************************************************************************************')
        print("Training model on CV : {} with indices  train: {} to {}".format(cv_count, idx_cv[0][0], idx_cv[0][-1]))
        print('                                         val: {} to {}'.format(idx_cv[1][0], idx_cv[1][-1]))
        print('********************************************************************************************')

        train_tuple = [X_train, y_train]
        val_tuple = [X_val, y_val]
        model = models.run_model(train_data=train_tuple, val_data=val_tuple, task_type='regression', model_type=type_of_model,
                                model_params=model_params, fit_params=fit_params, save_to_drive=save_to_drive,
                                save_folder=save_folder, legacy_save=legacy_save, cv_count=cv_count)

        if calculate_metrics:
            # predict
            predictions_train = predict_in_era_batch(model, train_data[feature_names], era_idx=era_idx_train,
                                                     rank_per_era=True)
            predictions_val = predict_in_era_batch(model, val_data[feature_names], era_idx=era_idx_val, rank_per_era=True)

            # save to dataframe
            train_data[PREDICTION_NAME] = predictions_train
            val_data[PREDICTION_NAME] = predictions_val

            # gather predictions
            predictions_train_total.append(predictions_train)
            predictions_val_total.append(predictions_val)

            # do the same for the whole tournament data
            if ((tour_df is not None) and (not tour_df.empty)):
                predictions_tour = predict_in_era_batch(model, tour_df[feature_names], era_idx=era_idx_tour,
                                                        rank_per_era=True)
                tour_df[PREDICTION_NAME] = predictions_tour
                predictions_tour_total.append(predictions_tour)

            # print metrics for this fold
            metrics = print_metrics_new(train_df=train_data, val_df=val_data, tour_df=tour_df,
                                        feature_names=feature_names, target_name=target_name,
                                        pred_name=PREDICTION_NAME)
            val_correlations, tour_correlations, val_sharpe, tour_sharpe = metrics

            if plot_metrics:
                plot_corrs_per_era_new(df=val_data, pred_name=PREDICTION_NAME, target_name=target_name)
                print(120 * '*')
                if ((tour_df is not None) and (not tour_df.empty)):
                    plot_corrs_per_era_new(df=tour_df, pred_name=PREDICTION_NAME, target_name=target_name)

            # average performance of each fold
            val_corrs_mean_cv.append(val_correlations.mean())
            val_corrs_std_cv.append(val_correlations.std(ddof=0))
            val_sharpe_cv.append(val_sharpe)

            if ((tour_df is not None) and (not tour_df.empty)):
                tour_corrs_mean_cv.append(tour_correlations.mean())
                tour_corrs_std_cv.append(tour_correlations.std(ddof=0))
                tour_sharpe_cv.append(tour_sharpe)
            else:
                # tour_correlations is already an empty list from the print_metrics_new function
                tour_sharpe_cv=None
        else:
            val_correlations, tour_correlations, val_sharpe_cv, tour_sharpe_cv = [0, 0, 0, 0]

    return val_correlations, tour_correlations, val_sharpe_cv, tour_sharpe_cv


def cv_split_creator(df, col, cv_scheme=TimeSeriesSplitGroups, n_splits=4, is_string=False,
                     extra_constructor_params={}, extra_fit_params={}, return_col=False):

    # add another column with date id to feed the cv splitter
    if col+'_No' not in df.columns:
        if is_string:
            dateno_values = [int(''.join(i for i in x if i.isdigit())) for x in df[col]]
            # dateno_values need to be pd.Series or pd.DataFrame
            dateno_values = pd.Series(dateno_values)
            df.insert(loc=1, column=col + '_No', value=dateno_values)
        else:
            dateno_values = df[col]
            df.insert(loc=1, column=col + '_No', value=dateno_values)
    else:
        dateno_values = df[col+'_No']

    # create TimeSeriesGroupSplit object and use .split to create our folds
    time_group_splitter = cv_scheme(n_splits=n_splits,
    **extra_constructor_params).split(df, groups=dateno_values, **extra_fit_params)

    # keep the data in list format
    cv_split_data = list(time_group_splitter)

    if not return_col:
        return cv_split_data
    else:
        return cv_split_data, dateno_values
