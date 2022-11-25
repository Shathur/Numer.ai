import os
import lightgbm as lgb
import xgboost as xgb


# get models into a list for iteration on them
def get_model_lst(num_models=1, prefix=None, folder_name=None, verbose=True):
    """

    Parameters
    ----------
    num_models: If 0 keep all the models in the folder
    prefix
    folder_name
    verbose: boolean 0 stay silent -- 1 print(models list)

    Returns
    -------

    """
    model_lst = [folder_name + x for x in os.listdir(folder_name)]
    if prefix is not None:
        model_lst = [x for x in model_lst if x.startswith(folder_name+prefix)]
    else:
        pass
    if num_models != 0:
        model_lst_final = model_lst[0:num_models]
    else:
        model_lst_final = model_lst
    if verbose:
        print(model_lst_final)

    return model_lst_final


def run_model(train_data=None, val_data=None, model_type='xgb', model_params=None, save_to_drive=False,
              save_folder=None, cv_count=None):
    X_train, y_train = train_data
    X_val, y_val = val_data

    model = create_model(model_type=model_type, model_params=model_params)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

    if save_to_drive:
        model.save_model(save_folder + 'model_{}.'.format(cv_count)+model_type)
        model.save_model(os.path.join(save_folder, f'model_{cv_count}.', model_type))
    else:
        model.save_model(os.path.join(f'model_{cv_count}.', model_type))

    return model


def create_model(model_type='xgb', model_params=None):
    if model_params is None:
        model_params = get_default_params(model_type=model_type)
    else:
        pass

    if model_type == 'lgb':
        model = lgb.LGBMRegressor()
        model.set_params(**model_params)

    if model_type == 'xgb':
        model = xgb.XGBRegressor()
        model.set_params(**model_params)

    return model


def get_default_params(model_type='xgb'):
    if model_type == 'lgb':
        params = {
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'num_leaves': 2 ** 5,
            'max_depth': 5,
            'colsample_bytree': 0.6,
            'device': "gpu",
        }
    elif model_type == 'xgb':
        params = {
            'max_depth': 5,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'n_jobs': -1,
            'colsample_bytree': 0.6,
            'tree_method': 'gpu_hist',
            'verbosity': 0
        }

    return params
