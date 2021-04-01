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