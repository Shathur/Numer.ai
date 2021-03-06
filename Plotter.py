import numerapi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

    return models_df_lst


a = compare(['melian', 'ulmo'], 250, 'corr+mmc')


class Plotter:
    def __init__(self, models_lst, round_id, metric_to_plot):
        self.models_lst = models_lst
        self.round_id = round_id
        self.metric_to_plot = metric_to_plot

    @staticmethod
    def fix_format(date):
        date = date.strftime('%b') + ' ' + date.strftime('%e')
        return date

    def model_daily_scores(self, model_id):
        model_daily = napi.daily_submissions_performances(model_id)
        model_daily_df = pd.DataFrame(model_daily)
        # add corr+mmc
        model_daily_df['corr+halfmmc'] = model_daily_df['correlation'] + 0.5*model_daily_df['mmc']
        model_daily_df['corr+mmc'] = model_daily_df['correlation'] + model_daily_df['mmc']
        model_daily_df['corr+2mmc'] = model_daily_df['correlation'] + 2*model_daily_df['mmc']
        model_round = model_daily_df[model_daily_df['roundNumber'].isin(self.round_id)].reset_index(drop=True)
        model_round['id'] = model_id
        return model_round

    def get_models_df_lst(self):
        models_df_lst = []
        for model_id in self.models_lst:
            model_round = self.model_daily_scores(model_id)
            models_df_lst.append(model_round)
        return models_df_lst

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
            ax.set_xlim((0, model_rnd.shape[0]//len(self.models_lst)))
            sns.lineplot(ax=ax, x='date', y='value', hue='id', data=models_melted, sort=False)  # hue='variable
            plt.show()

            count_rnd += 1

    def create_subplot_axes(self):
        cols_num = 3
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


def read_model():
    stored_csv = pd.read_csv('C:/Users/Ilias/Desktop/Numer.ai/Changes_Lookup_Destination.csv')
    return stored_csv

xa = read_model()
xaxa = xa[xa['MODELS']=='melian'].dropna(axis=1)
change_info = xaxa.iloc[:, -1:].iloc[0,0].split(',')

prev_slot = xa[xa['MODELS']==change_info[2]].dropna(axis=1)
prev_slot[prev_slot.iloc[0, :].str.contains('melian')]
prev_slot.loc[:, column.isin(['CHANGE1', 'CHANGE2'])].str.contains('lala')




plotter = Plotter(['melian', 'ulmo'], [250, 251, 255, 256, 263], 'corr+mmc')
plotter.plot_subplot()

compare(['melian', 'ulmo'], 250, 'corr+mmc')

model_daily = napi.daily_submissions_performances('melian')
model_daily_df = pd.DataFrame(model_daily)
model_daily_df.dtypes

for model_id in ['melian', 'ulmo']:
    model_daily = napi.daily_submissions_performances(model_id)
    model_daily_df = pd.DataFrame(model_daily)
    # add corr+mmc
    model_daily_df['corr+mmc'] = model_daily_df['correlation'] + model_daily_df['mmc']
    model_round = model_daily_df[model_daily_df['roundNumber'] == 250].reset_index(drop=True)
    model_round['id'] = model_id

model_round

