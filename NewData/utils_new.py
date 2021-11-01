from basic_functions import *
from feature_neutralization import *

from basic_functions import print_metrics


# plot the correlations per era
# era columns in the new data include
# just the numbers
def plot_corrs_per_era_new(df=None, pred_name=None, target_name=TARGET_NAME, legend_title=None):
    val_corrs = corr_score(df, pred_name, target_name)
    plt.figure()
    ax = sns.barplot(x=val_corrs.index, y=val_corrs)
    if legend_title is not None:
        ax.legend(title=legend_title)
    plt.show()


def print_metrics_new(train_df=None, val_df=None, tour_df=None, feature_names=None,
                      pred_name=None, target_name=TARGET_NAME, long_metrics=True):
    """
    When you print neutralized metrics train_df has to be None cause we don't
    neutralize our targets on train_df
    feature_names : the columns of the features used. Used only when long_metrics=True,
                    otherwise can skip
    """

    if train_df is not None:
        # Check the per-era correlations on the training set (in sample)
        train_correlations = corr_score(train_df, pred_name, target_name)
        print(
            f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
    else:
        train_correlations = []

    if val_df is not None:
        '''Out of Fold Validation'''
        # Check the per-era correlations on the oof set (out of fold)
        oof_correlations = corr_score(val_df, pred_name, target_name)
        print(f"On oof the correlation has mean {oof_correlations.mean()} and "
              f"std {oof_correlations.std(ddof=0)}")
    else:
        oof_correlations = []

    if tour_df is not None:
        validation_correlations = corr_score(tour_df, pred_name, target_name)
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
        feature_exposures = tour_df[feature_names].apply(
            lambda d: spearman(tour_df[PREDICTION_NAME], d),
            axis=0)
        max_per_era = tour_df.groupby("era").apply(
            lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
        max_feature_exposure = max_per_era.mean()
        print(f"Max Feature Exposure for validation: {max_feature_exposure}")

        if val_df is not None:
            # Check feature neutral mean for oof
            feature_neutral_mean = get_feature_neutral_mean(val_df)
            print(f"Feature Neutral Mean for oof is {feature_neutral_mean}")

        # Check feature neutral mean for validation
        feature_neutral_mean = get_feature_neutral_mean(tour_df)
        print(f"Feature Neutral Mean for validation is {feature_neutral_mean}")

    return [oof_correlations, validation_correlations, oof_sharpe, validation_sharpe]


# Feature Neutralization and plot the results
def plot_feature_neutralization_new(tour_df, neut_percent, full=False,
                                    feature_names=None, target_name=TARGET_NAME, show_metrics=False,
                                    legend_title=None):
    # Will be depracated in future versions and changed with new data validation scheme
    if not full:
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

    val_corrs_neut = corr_score(validation_data, PREDICTION_NAME_NEUTRALIZED, target_name)
    val_sharpe_neut = sharpe_score(val_corrs_neut)

    # Plot the feature exposures with neutralization per era
    validation_data[PRED_NAME_NEUT_PER_ERA] = neutralize(
        df=validation_data, columns=[PREDICTION_NAME],
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

    # plot and print correlations per era
    if show_metrics:
        metrics = print_metrics_new(tour_df=validation_data, pred_name=PREDICTION_NAME,
                                    feature_names=feature_names,
                                    long_metrics=True)

    plot_corrs_per_era_new(validation_data, PREDICTION_NAME, legend_title[0])

    if show_metrics:
        metrics = print_metrics(tour_df=validation_data,
                                pred_name=PREDICTION_NAME_NEUTRALIZED,
                                feature_names=feature_names,
                                long_metrics=True)

    plot_corrs_per_era_new(validation_data, PREDICTION_NAME_NEUTRALIZED, legend_title[1])

    if show_metrics:
        metrics = print_metrics_new(tour_df=validation_data,
                                    pred_name=PRED_NAME_NEUT_PER_ERA,
                                    feature_names=feature_names,
                                    long_metrics=True)

    plot_corrs_per_era_new(validation_data, PRED_NAME_NEUT_PER_ERA, legend_title[2])
