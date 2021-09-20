from basic_functions import *
from feature_neutralization import *


# plot the correlations per era
# era columns in the new data include
# just the numbers
def plot_corrs_per_era_new(df, pred_name):
    val_corrs = corr_score(df, pred_name)
    plt.figure()
    ax = sns.barplot(x=val_corrs.index, y=val_corrs)
    plt.show()


def print_metrics_new(train_df=None, val_df=None, tour_df=None, feature_names=None, pred_name=None, long_metrics=True,
                      scores_on_val2=False):
    # when you print neutralized metrics train_df has to be None cause we don't
    # neutralize our targets on train_df

    if train_df is not None:
        # Check the per-era correlations on the training set (in sample)
        train_correlations = corr_score(train_df, pred_name)
        print(
            f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
    else:
        train_correlations = []

    if val_df is not None:
        '''Out of Fold Validation'''
        # Check the per-era correlations on the oof set (out of fold)
        oof_correlations = corr_score(val_df, pred_name)
        print(f"On oof the correlation has mean {oof_correlations.mean()} and "
              f"std {oof_correlations.std(ddof=0)}")
    else:
        oof_correlations = []

    if tour_df is not None:
        validation_correlations = corr_score(tour_df, pred_name)
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