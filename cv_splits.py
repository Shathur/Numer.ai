import numpy as np

from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples


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