import numpy as np

from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples

from sklearn.metrics import log_loss, accuracy_score


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
            yield (indices[groups.isin(group_lst[i*cutoff_eras: i*cutoff_eras + cutoff_eras])],
                   indices[groups.isin(group_lst[i*cutoff_eras: i*cutoff_eras + cutoff_eras])])


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
    def __init(self, n_splits=None, t1=None, pctEmbargo=None):
        super(PurgedKfold, self).__init(n_splits, shuffle=False, random_state=None)
        self.t1=t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0].self.pctEmbargo)
        test_starts = [(i[0], i[-1]+1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]

        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i: j]
            maxt1idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            if maxt1idx<X.shape[0]:
                train_indices = np.concatenate((train_indices, indices[maxt1idx+mbrg:]))
            yield train_indices, test_indices


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
        fit=clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob=fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)

        score.append(score_)

    return np.array(score)


