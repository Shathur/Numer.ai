import pandas as pd
import numpy as np

import basic_functions as bf

import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm.notebook import tqdm
import os
import gc


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


# predict in batches. xgb and lgb supported only atm
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


# predict in batches on a per-era basis. xgb and lgb supported only atm
def get_predictions_per_era(df=None, num_models=1, prefix=None, folder_name=None, era_idx=[],
                            model_type='xgb', rank_average=False):
    """

    :param df: dataframe with the features used to train and predict
    :param num_models: number of models in the folder
    :param prefix: prefix to choose specific models from the folder - use it only if you had run a CV scheme
                   for many different targets or something
    :param folder_name: name of the folder
    :param era_idx: indices of dataframe
    :param model_type: xgb or lgb
    :param rank_average: True - rank the predictions per era or False -  total ranks in the whole dataframe
    :return: final predictions with proper dimensions for further use
    """
    model_lst = bf.get_model_lst(num_models=num_models, prefix=prefix, folder_name=folder_name)
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


def get_predictions_per_era_joblib(df, preds_cache_file=None, num_models=1, prefix=None, len_live=None,
                                   era_idx=[], era_x_idx=[], model_type='xgb', folder_name=None,
                                   rank_average=False, first_week=False):
    """

    Parameters
    ----------
    df: dataframe with the proper columns for predictions
    preds_cache_file: saved dictionary with the format {num_of_models_predicted: predictions}
    num_models: number of models in the folder to ensemble predictions
    prefix: prefix to separate predictions from different training folds eg. 'target_nomi_'
    len_live: the length of live data. When first_week=False it loads the len_live from last week
    era_idx: indices of each era
    era_x_idx: indices of eraX. If not predicting live data leave blank
    model_type: choose between xgb or lgb
    folder_name: folder where models are saved
    rank_average: True - rank the predictions per era or False -  total ranks in the whole dataframe
    first_week: boolean

    Returns
    -------
    predictions: final predictions with proper format

    """
    first_time_new_week = True

    if os.path.isfile(preds_cache_file):
        with open(preds_cache_file, 'rb') as file:
            cache = pickle.load(file)
        file.close()
    else:
        cache = {-1: []}

    model_lst = bf.get_model_lst(num_models=num_models, prefix=prefix, folder_name=folder_name)
    predictions_total = []
    predictions_total_era_x = []
    for cv_num in tqdm(range(num_models)):
        if cv_num > int(list(cache)[0]):  # check if the model predictions have been saved
            if model_type == 'lgb':
                model = lgb.Booster(model_file=model_lst[cv_num])
            if model_type == 'xgb':
                model = bf.create_model(model_type='xgb')
                model.load_model(model_lst[cv_num])

            predictions = predict_in_era_batch(model=model,
                                               df=df,
                                               era_idx=era_idx,
                                               rank_per_era=rank_average)

            if rank_average:
                # predictions are ranks if rank_per_era=True. In every iteration
                # we gather the predictions that we normalize and the previous
                # ensembling of prediction which is already normalized.
                # So we np.array them to transform them into a matrix, we reshape
                # given matrix with .reshape(-1, 1), we .squeeze() the result to bring
                # the shape back to our original one and lastly we transform them
                # back into a list
                scaler = MinMaxScaler(feature_range=(0, 1))
                preds = scaler.fit_transform(X=np.array(predictions).reshape(-1, 1)).squeeze().tolist()
                predictions = preds

            predictions_total.append(predictions)
            # print(predictions_total[cv_num][0:10])
            # print(predictions_total[1][0:10])

            predictions_final = np.sum(predictions_total, axis=0)

            """
            # save as dictionary. {num_of_aggregated: predictions}
            # format of the predictions is similar to get_predictions_per_era function
            """
            if len_live:
                cache = {cv_num: predictions_final,
                        'len_live': len_live}
            else:
                cache = {cv_num: predictions_final}

            # print(dict(itertools.islice(cache.items(), 10)))
            # print([value for (key, value) in cache.items()][0][0:10])

            with open(preds_cache_file, 'wb') as file:
                pickle.dump(cache, file)
            file.close()

        else:
            # if loading file, create list then load the aggregated predictions
            # we keep the values of the dict
            predictions_total = []
            predictions_total.append([value for (key, value) in cache.items()][0]) #.tolist()

            if len_live:
                len_live = [value for (key, value) in cache.items()][1]

            if era_x_idx:
                # in every model that we have already predicted, we still predict
                # for eraX. We keep everything else the same and we ensemble the predictions
                # for eraX for all models. At the end of each iteration we update
                # the ensemble only for the eraX part of the predictions.
                if model_type == 'lgb':
                    model = lgb.Booster(model_file=model_lst[cv_num])
                if model_type == 'xgb':
                    model = bf.create_model(model_type='xgb')
                    model.load_model(model_lst[cv_num])

                predictions_era_x = predict_in_era_batch(model=model,
                                                         df=df,
                                                         era_idx=era_x_idx,
                                                         rank_per_era=True)

                if rank_average:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    preds = scaler.fit_transform(X=np.array(predictions_era_x).reshape(-1, 1)).squeeze().tolist()
                    predictions_era_x = preds

                predictions_total_era_x.append(predictions_era_x)

                predictions_final_era_x = np.sum(predictions_total_era_x, axis=0)

                # update the cached list with the eraX predictions
                if first_week:
                    # update only the eraX predictions from the cached list
                    predictions_total[0][-len(predictions_final_era_x):] = predictions_final_era_x
                else:
                    if first_time_new_week:
                        # predictions_total[0].tolist().extend(predictions_final_era_x)
                        predictions_old = predictions_total[0][: -len_live]
                        predictions_total[0] = predictions_old
                        predictions_total[0].tolist().extend(predictions_final_era_x)
                        predictions_total[0] = predictions_total[0].tolist().extend(predictions_final_era_x)
                        # predictions_total[0] = np.array(predictions_total[0].tolist().extend(predictions_final_era_x))
                        first_time_new_week = False
                    else:
                        # update only the eraX predictions from the cached list
                        predictions_total[0][-len(predictions_final_era_x):] = predictions_final_era_x

                # save the already calculated test predictions with the so far
                # averaged predictions of eraX for models till model no cv_num
                # save as dictionary. {num_of_aggregated: predictions}
                # format of the predictions is similar to get_predictions_per_era function
                cache = {list(cache.keys())[0]: predictions_total[0], # list(cache.keys())[0]: np.array(predictions_total[0])
                         list(cache.keys())[1]: len(predictions_final_era_x)}
                with open(preds_cache_file, 'wb') as file:
                    pickle.dump(cache, file)
                file.close()

    if rank_average:
        scaler = MinMaxScaler(feature_range=(0, 1))
        predictions_final = scaler.fit_transform(X=(np.sum(predictions_total, axis=0) / num_models).reshape(-1, 1))
    else:
        predictions_final = np.sum(predictions_total, axis=0) / num_models

    gc.collect()

    return predictions_final.squeeze()