from utils import *
from feature_neutralization import *
import utils
import feature_neutralization as fn


# plot the correlations per era
# era columns in the new data include
# just the numbers
def plot_corrs_per_era_new(df=None, pred_name='prediction', target_name='target', legend_title=None):
    val_corrs = utils.corr_score(df, pred_name, target_name)
    plt.figure()
    ax = sns.barplot(x=val_corrs.index, y=val_corrs)
    if legend_title is not None:
        ax.legend(title=legend_title)
    plt.xticks(rotation=45)
    plt.show()


def print_metrics_new(train_df=None, val_df=None, tour_df=None, feature_names=None,
                      pred_name='prediction', target_name='target', long_metrics=True):
    """
    When you print neutralized metrics train_df has to be None cause we don't
    neutralize our targets on train_df
    feature_names : the columns of the features used. Used only when long_metrics=True,
                    otherwise can skip
    """

    if train_df is not None:
        # Check the per-era correlations on the training set (in sample)
        train_correlations = utils.corr_score(train_df, pred_name, target_name)
        print(
            f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
    else:
        train_correlations = []

    if val_df is not None:
        '''Out of Fold Validation'''
        # Check the per-era correlations on the oof set (out of fold)
        oof_correlations = utils.corr_score(val_df, pred_name, target_name)
        print(f"On oof the correlation has mean {oof_correlations.mean()} and "
              f"std {oof_correlations.std(ddof=0)}")
    else:
        oof_correlations = []

    if tour_df is not None:
        validation_correlations = utils.corr_score(tour_df, pred_name, target_name)
        print(f"On validation the correlation has mean {validation_correlations.mean()} and "
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
        validation_sharpe = utils.sharpe_score(validation_correlations)
        print(f"Validation Sharpe: {validation_sharpe}")
    else:
        validation_sharpe = []

    if long_metrics:
        if val_df is not None:
            # Checking the max drowdown of the oof
            rolling_max = (oof_correlations + 1).cumprod().rolling(window=100,
                                                                   min_periods=1).max()
            daily_value = (oof_correlations + 1).cumprod()
            max_drawdown = -(rolling_max - daily_value).max()
            print(f"max drawdown oof: {max_drawdown}")

        if tour_df is not None:
            # checking the max drowdown of the validation
            rolling_max = (validation_correlations + 1).cumprod().rolling(window=100,
                                                                          min_periods=1).max()
            daily_value = (validation_correlations + 1).cumprod()
            max_drawdown = -(rolling_max - daily_value).max()
            print(f"max drawdown validation: {max_drawdown}")

        if val_df is not None:
            # Check the feature exposure of your oof predictions
            # feature_exposures = val_df[feature_names].apply(lambda d: utils.spearman(val_df[pred_name], d),
            #                                                 axis=0)
            max_per_era = val_df.groupby("era").apply(
                lambda d: d[feature_names].corrwith(d[pred_name]).abs().max())
            max_feature_exposure = max_per_era.mean()
            print(f"Max Feature Exposure for oof: {max_feature_exposure}")

        if tour_df is not None:
            # Check the feature exposure of your validation predictions
            # feature_exposures = tour_df[feature_names].apply(
            #     lambda d: utils.spearman(tour_df[pred_name], d),
            #     axis=0)
            max_per_era = tour_df.groupby("era").apply(
                lambda d: d[feature_names].corrwith(d[pred_name]).abs().max())
            max_feature_exposure = max_per_era.mean()
            print(f"Max Feature Exposure for validation: {max_feature_exposure}")

        if val_df is not None:
            # Check feature neutral mean for oof
            feature_neutral_mean = utils.get_feature_neutral_mean(val_df, pred_name, target_name)
            print(f"Feature Neutral Mean for oof is {feature_neutral_mean}")

        if tour_df is not None:
            # Check feature neutral mean for validation
            feature_neutral_mean = utils.get_feature_neutral_mean(tour_df, pred_name, target_name)
            print(f"Feature Neutral Mean for validation is {feature_neutral_mean}")

    return [oof_correlations, validation_correlations, oof_sharpe, validation_sharpe]


# FN on either tournament or validation data
def run_feature_neutralization_new(df=None, predictions_total=None, target_name='target',
                                   pred_name='prediction',
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
        val_corrs = utils.corr_score(validation_data, pred_name, target_name)
        sharpe = utils.sharpe_score(val_corrs)
        print('Validation correlations : {}\n'
              'Validation sharpe : {}'.format(val_corrs.mean(), sharpe))
        metrics = print_metrics_new(tour_df=df, pred_name=pred_name, target_name=target_name,
                                    long_metrics=False)

        # run only for FN

        # DEFINE FEATURE NEUTRALIZATION PERCENTAGE
        # CAREFUL THIS IS DIFFERENT BETWEEN SUBMISSIONS
        # neut_type has to be either 'short' for neutralize_short()
        #                     either 'perera' for neutralize()
        if neut_type == 'short':
            df[pred_name + '_neutralized'] = fn.neutralize_short(df,
                                                                 prediction_name=pred_name,
                                                                 proportion=proportion)
        elif neut_type == 'perera':
            df[pred_name + '_neutralized'] = fn.neutralize(df=df, columns=[pred_name],
                                                           extra_neutralizers=df.columns[
                                                               df.columns.str.startswith('feature')],
                                                           proportion=proportion, normalize=True, era_col='era')

        validation_data = df[df['data_type'] == 'validation']
        val_corrs = utils.corr_score(validation_data, pred_name + '_neutralized', target_name)
        sharpe = utils.sharpe_score(val_corrs)
        print('Validation correlations : {}\n'
              'Validation sharpe : {}'.format(val_corrs.mean(), sharpe))

        # metrics will differ somewhat from training notebook cause here we neutralized the whole tournament_data
        # for submission purposes, while in the training notebook we neutralize only the validation_data.
        metrics = print_metrics_new(tour_df=df, pred_name=pred_name + '_neutralized', target_name=target_name,
                                    long_metrics=False)

        # Rescale into [0,1] range keeping rank
        minmaxscaler = MinMaxScaler(feature_range=(0, 0.999999))
        minmaxscaler.fit(np.array(df[pred_name + '_neutralized']).reshape(-1, 1))
        df[pred_name + '_neutralized'] = minmaxscaler.transform(
            np.array(df[pred_name + '_neutralized']).reshape(-1, 1))

        # preds = df[pred_name+'_neutralized'].copy()
        preds = df[pred_name + '_neutralized'].values  # np.array of predictions
    return preds


# Feature Neutralization and plot the results
def plot_feature_neutralization_new(tour_df, neut_percent, full=False,
                                    feature_names=None,
                                    pred_name='prediction',
                                    target_name='target',
                                    plot_feature_exposures=True,
                                    neutralize_total=False,
                                    neutralize_per_era=True,
                                    show_metrics=False,
                                    legend_title=None):
    # Will be depracated in future versions and changed with new data validation scheme
    if not full:
        validation_data = tour_df[tour_df.data_type == "validation"].copy()
    else:
        val2_eras = list(range(197, 213))
        val2_eras = ['era' + str(x) for x in val2_eras]
        validation_data = tour_df[tour_df['era'].isin(val2_eras)]

    if plot_feature_exposures:
        # Plot feature exposures
        feat_exps, feats = utils.feature_exposures(validation_data, pred_name)

        plt.figure()
        ax = sns.barplot(x=feats, y=feat_exps)
        ax.legend(title='Max_feature_exposure : {}\n'
                        'Mean feature exposure : {}'.format(utils.max_feature_exposure(validation_data, pred_name),
                                                            utils.feature_exposure(validation_data, pred_name)))
        plt.show()

    if neutralize_total:
        # Plot the feature exposures with neutralization
        validation_data[pred_name + '_neutralized'] = fn.neutralize_short(validation_data,
                                                                          prediction_name=pred_name,
                                                                          proportion=neut_percent)

        feat_exps, feats = utils.feature_exposures(validation_data, pred_name + '_neutralized')

        plt.figure()
        ax1 = sns.barplot(x=feats, y=feat_exps)
        ax1.legend(title='Max_feature_exposure : {}\n'
                         'Mean feature exposure : {}'.format(
            utils.max_feature_exposure(validation_data, pred_name + '_neutralized'),
            utils.feature_exposure(validation_data, pred_name + '_neutralized')))
        plt.show()

        val_corrs_neut = utils.corr_score(validation_data, pred_name + '_neutralized', target_name)
        val_sharpe_neut = utils.sharpe_score(val_corrs_neut)

    if neutralize_per_era:
        # Plot the feature exposures with neutralization per era
        validation_data[PRED_NAME_NEUT_PER_ERA] = fn.neutralize(
            df=validation_data, columns=[pred_name],
            extra_neutralizers=feature_names,
            proportion=neut_percent, normalize=True, era_col='era')

        # validation_data[PRED_NAME_NEUT_PER_ERA] = minmax_scale_values(df=validation_data,
        #                                                               pred_name=PRED_NAME_NEUT_PER_ERA)

        feat_exps, feats = utils.feature_exposures(validation_data, PRED_NAME_NEUT_PER_ERA)

        plt.figure()
        ax1 = sns.barplot(x=feats, y=feat_exps)
        ax1.legend(title='Max_feature_exposure : {}\n'
                         'Mean feature exposure : {}'.format(
            utils.max_feature_exposure(validation_data, PRED_NAME_NEUT_PER_ERA),
            utils.feature_exposure(validation_data, PRED_NAME_NEUT_PER_ERA)))
        plt.show()

    # plot and print correlations per era
    if show_metrics:
        metrics = print_metrics_new(tour_df=validation_data, pred_name=pred_name,
                                    feature_names=feature_names,
                                    long_metrics=True)

    if legend_title is not None:
        plot_corrs_per_era_new(validation_data, pred_name, legend_title[0])
    else:
        plot_corrs_per_era_new(validation_data, pred_name)

    if neutralize_total:
        if show_metrics:
            metrics = print_metrics_new(tour_df=validation_data,
                                        pred_name=pred_name + '_neutralized',
                                        feature_names=feature_names,
                                        long_metrics=True)
        if legend_title is not None:
            plot_corrs_per_era_new(validation_data, pred_name + '_neutralized', legend_title[1])
        else:
            plot_corrs_per_era_new(validation_data, pred_name + '_neutralized')

    if neutralize_per_era:
        if show_metrics:
            metrics = print_metrics_new(tour_df=validation_data,
                                        pred_name=PRED_NAME_NEUT_PER_ERA,
                                        feature_names=feature_names,
                                        long_metrics=True)
        if legend_title is not None:
            plot_corrs_per_era_new(validation_data, PRED_NAME_NEUT_PER_ERA, legend_title[2])
        else:
            plot_corrs_per_era_new(validation_data, PRED_NAME_NEUT_PER_ERA)
