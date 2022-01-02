import pandas as pd
import numpy as np

import basic_functions as bf

import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler


# predict in batches to avoid memory issues
# df should contain only the prediction features
# i.e. training_data[feature_names]
def predict_in_batch(model, df, batch):
    predictions = []
    for i in range(0, len(df), batch):
        preds = model.predict(df[i: i + batch])
        predictions.extend(preds)
    return predictions


# predict in era batches to avoid memory issues
# df should contain only the prediction features
# i.e. training_data[feature_names]
def predict_in_era_batch(model, df, era_idx, rank_per_era):
    """
    rank_per_era=True : returns predictions that are the ranking
        of our predictions per era eg.[2,5,4,9,1,6,7,8,3,0]
    rank_per_era=False : returns the predictions of each era,
        concatenated to one another
    """
    predictions = []
    for era in era_idx:
        preds = model.predict(df.loc[era])
        if rank_per_era:
            preds = np.array(pd.Series(preds).rank())
        predictions.extend(preds)
    return predictions


# predict in batches. XGBRegressor supported only atm
def get_predictions(df=None, num_models=1, prefix=None, folder_name=None, model_type='xgb', batch_size=20000):
    """

    :param df: dataframe with the features used to train and predict
    :param num_models: number of models in the folder
    :param prefix: prefix to choose specific models from the folder - use it only if you had run a CV scheme
                   for many different targets or something
    :param folder_name: name of the folder
    :param model_type: xgb or lgb
    :param batch_size: predict in batch_size equal to this number
    :return: np.array with predictions for the df
    """
    model_lst = bf.get_model_lst(num_models=num_models, prefix=prefix, folder_name=folder_name)
    predictions_total = []
    for cv_num in range(num_models):
        if model_type == 'lgb':
            model = lgb.Booster(model_file=model_lst[cv_num])
        if model_type == 'xgb':
            model = bf.create_model(model_type='xgb')
            model.load_model(model_lst[cv_num])

        X_test = df

        predictions = predict_in_batch(model, X_test, batch_size)

        predictions_total.append(predictions)

    predictions_total = np.mean(predictions_total, axis=0)

    return predictions_total


# predict in batches. xgb and lgb supported only atm
def get_predictions_per_era(df=None, num_models=1, folder_name=None, era_idx=[],
                            model_type='xgb', rank_average=False):
    """

    :param df: dataframe with the features used to train and predict
    :param num_models: number of models in the folder
    :param folder_name: name of the folder
    :param era_idx: indices of dataframe
    :param model_type: xgb or lgb
    :param rank_average: True - rank the predictions per era or False -  total ranks in the whole dataframe
    :return: final predictions with proper dimensions for further use
    """
    model_lst = bf.get_model_lst(num_models=num_models, folder_name=folder_name)
    predictions_total = []

    X_test = df

    for cv_num in range(num_models):
        if model_type == 'lgb':
            model = lgb.Booster(model_file=model_lst[cv_num])
        if model_type == 'xgb':
            model = bf.create_model(model_type='xgb')
            model.load_model(model_lst[cv_num])

        predictions = predict_in_era_batch(model=model,
                                           df=X_test,
                                           era_idx=era_idx,
                                           rank_per_era=rank_average)

        predictions_total.append(predictions)

    if rank_average:
        scaler = MinMaxScaler(feature_range=(0, 1))
        predictions_final = scaler.fit_transform(X=np.mean(predictions_total, axis=0).reshape(-1, 1))
    else:
        predictions_final = np.mean(predictions_total, axis=0)

    return predictions_final.squeeze()