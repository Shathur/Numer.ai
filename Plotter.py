import numerapi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime


# set size of plots
# sns.set(rc={'figure.figsize' : (15, 8.27)})


def fix_format(date):
    date = date.strftime('%b') + ' ' + date.strftime('%e')
    return date


napi = numerapi.NumerAPI()


def compare(models_lst, round, metric_to_plot):
    models_df_lst = []
    for model_id in models_lst:
        model_daily = napi.daily_submissions_performances(model_id)
        model_daily_df = pd.DataFrame(model_daily)
        # add corr+mmc
        model_daily_df['corr+mmc'] = model_daily_df['correlation'] + model_daily_df['mmc']
        model_round = model_daily_df[model_daily_df['roundNumber'] == round].reset_index(drop=True)
        model_round['id'] = model_id
        models_df_lst.append(model_round)

    models_melted = pd.melt(pd.concat(models_df_lst).reset_index(drop=True), id_vars=['id', 'date'],
                            value_vars=[metric_to_plot])
    models_melted['date'] = models_melted['date'].apply(fix_format)

    plt.figure()
    sns.lineplot(x='date', y='value', hue='id', data=models_melted, sort=False)  # hue='variable
    plt.xticks(rotation=45)
    plt.xlim(0)
    plt.show()

    return models_df_lst


def compare(models_lst, round, metric_to_plot):
    models_df_lst = []
    for model_id in models_lst:
        model_daily = napi.daily_submissions_performances(model_id)
        model_daily_df = pd.DataFrame(model_daily)
        # add corr+mmc
        model_daily_df['corr+mmc'] = model_daily_df['correlation'] + model_daily_df['mmc']
        model_round = model_daily_df[model_daily_df['roundNumber'] == round].reset_index(drop=True)
        model_round['id'] = model_id
        models_df_lst.append(model_round)

    models_melted = pd.melt(pd.concat(models_df_lst).reset_index(drop=True), id_vars=['id', 'date'],
                            value_vars=[metric_to_plot])
    models_melted['date'] = models_melted['date'].apply(fix_format)

    fig, axes = plt.subplots(1, squeeze=False)
    sns.lineplot(ax=axes[0], x='date', y='value', hue='id', data=models_melted, sort=False)  # hue='variable
    plt.xticks(rotation=45)
    plt.xlim(0)
    plt.show()

    return models_df


class Plotter:
    def __init__(self, models_lst, round_id, metric_to_plot, horizontal_columns):
        self.models_lst = models_lst
        self.round_id = round_id
        self.metric_to_plot = metric_to_plot
        self.horizontal_columns = horizontal_columns

    @staticmethod
    def fix_format(date):
        date = date.strftime('%b') + ' ' + date.strftime('%e')
        return date

    def model_daily_scores(self, model_id):
        model_daily = napi.daily_submissions_performances(model_id)
        model_daily_df = pd.DataFrame(model_daily)
        # add corr+mmc
        model_daily_df['corr+halfmmc'] = model_daily_df['correlation'] + 0.5 * model_daily_df['mmc']
        model_daily_df['corr+mmc'] = model_daily_df['correlation'] + model_daily_df['mmc']
        model_daily_df['corr+2mmc'] = model_daily_df['correlation'] + 2 * model_daily_df['mmc']
        model_round = model_daily_df[model_daily_df['roundNumber'].isin(self.round_id)].reset_index(drop=True)
        model_round['id'] = model_id
        return model_round

    def get_models_df_lst(self):
        models_df_lst = []
        for model_id in self.models_lst:
            model_round = self.model_daily_scores(model_id)
            models_df_lst.append(model_round)
        return models_df_lst

    # def daily_scores(self, model_id, rnd):
    #     model_daily = napi.daily_submissions_performances(model_id)
    #     model_daily_df = pd.DataFrame(model_daily)
    #     # add corr+mmc
    #     model_daily_df['corr+halfmmc'] = model_daily_df['correlation'] + 0.5*model_daily_df['mmc']
    #     model_daily_df['corr+mmc'] = model_daily_df['correlation'] + model_daily_df['mmc']
    #     model_daily_df['corr+2mmc'] = model_daily_df['correlation'] + 2*model_daily_df['mmc']
    #     model_round = model_daily_df[model_daily_df['roundNumber'] == rnd].reset_index(drop=True)
    #     model_round['id'] = model_id
    #     return model_round
    #
    # def get_models_df_mixed_lst(self):
    #     models_df_lst = []
    #     for model_rnd in self.models_rounds:
    #         model_round = self.model_daily_scores(model_rnd[0], model_rnd[1])
    #         models_df_lst.append(model_round)

    def plot_single_round(self):
        models_melted = pd.melt(pd.concat(self.get_models_df_lst()).reset_index(drop=True), id_vars=['id', 'date'],
                                value_vars=[self.metric_to_plot])
        models_melted['date'] = models_melted['date'].apply(self.fix_format)

        plt.figure()
        sns.lineplot(x='date', y='value', hue='id', data=models_melted, sort=False)  # hue='variable
        plt.xticks(rotation=45)
        plt.xlim(0)
        plt.show()

    def plot_subplot(self):
        count_rnd = 0
        models_df_lst = self.get_models_df_lst()
        fig, axes = self.create_subplot_axes()
        print(axes)
        for rnd in self.round_id:
            model_rnd = pd.concat(models_df_lst).reset_index(drop=True)
            model_rnd = model_rnd[model_rnd['roundNumber'] == rnd]
            models_melted = pd.melt(model_rnd, id_vars=['id', 'date'],
                                    value_vars=[self.metric_to_plot])
            models_melted['date'] = models_melted['date'].apply(self.fix_format)

            ax = axes.flatten()[count_rnd]
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.grid(True, axis='both')
            ax.set_xlim((0, model_rnd.shape[0] // len(self.models_lst)))
            sns.lineplot(ax=ax, x='date', y='value', hue='id', data=models_melted, sort=False)  # hue='variable
            plt.show()

            count_rnd += 1

    # def plot_subplot_mixed(self):
    #     models_df_lst = self.get_models_df_lst()
    #     fig, axes = self.create_subplot_axes()
    #     print(axes)
    #     for model_rnd in self.models_rounds:

    def create_subplot_axes(self):
        cols_num = self.horizontal_columns
        if len(self.round_id) % cols_num == 0:
            x_dim = len(self.round_id) // cols_num
            y_dim = cols_num
        else:
            x_dim = len(self.round_id) // cols_num + 1
            y_dim = cols_num
        # axes = np.ndarray(shape=(x_dim, y_dim))
        fig, axes = plt.subplots(x_dim, y_dim, figsize=(24, 6), squeeze=False)

        # flatten axes dimension to be used as positional arguments

        return fig, axes

    def plot_cv_split(self, df, cv_split_data, dateno_values, cv_scheme, target='target', ax=None, n_splits=4, lw=20):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        cmap_cv = plt.cm.coolwarm
        jet = plt.cm.get_cmap('jet', 256)
        seq = np.linspace(0, 1, 256)
        np.random.shuffle(seq)
        cmap_data = ListedColormap(jet(seq))
        for ii, (tr, tt) in enumerate(cv_split_data):
            indices = np.array([np.nan] * len(df))
            indices[tt] = 1
            indices[tr] = 0
            ax.scatter(range(len(indices)), [ii + .5] * len(indices), c=indices, marker='_', lw=lw, cmap=cmap_cv,
                       vmin=-.2, vmax=1.2)
        ax.scatter(range(len(df)), [ii + 1.5] * len(df), c=df[target], marker='_', lw=lw, cmap=plt.cm.Set3)
        ax.scatter(range(len(df)), [ii + 2.5] * len(df), c=np.array(dateno_values), marker='_', lw=lw, cmap=cmap_data)
        yticklabels = list(range(n_splits)) + [target, 'day']
        ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels, xlabel='Sample index',
               ylabel="CV iteration", ylim=[n_splits + 2.2, -.2], xlim=[0, len(df[target])])
        ax.set_title('{}'.format(cv_scheme.__name__), fontsize=15)
        return ax
