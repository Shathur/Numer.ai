import os
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, XGBClassifier
import json


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
    # model_lst = [folder_name + x for x in os.listdir(folder_name)]
    model_lst = [os.path.join(folder_name, x) for x in os.listdir(os.path.join(folder_name))]
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


def run_model(
    train_data=None,
    val_data=None,
    model_type='xgb',
    task_type='regression',
    model_params=None,
    fit_params=None,
    save_to_drive=False, 
    save_folder=None,
    legacy_save=True,
    cv_count=None,
):
    X_train, y_train = train_data
    X_val, y_val = val_data

    model = create_model(
        model_type=model_type,
        task_type=task_type,
        model_params=model_params,
    )

    if fit_params is None:
        fit_params = {
            'early_stopping_rounds': 10,
            'verbose': False
        }
    else:
        pass

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        **fit_params,
    )

    # if we are using lgb we first need to keep the booster of our model
    if model_type == 'lgb':
        model = model.booster_

    if save_to_drive:
        if legacy_save:
            if model_type=='lgb':
                model.save_model(os.path.join(save_folder,f'model_{cv_count}.{model_type}'))
            else:
                model.save_model(save_folder + 'model_{}.'.format(cv_count)+model_type)
        else:
            if model_type=='lgb':
                model_json = model.dump_model()
                with open(os.path.join(save_folder,f'model_{cv_count}.{model_type}'),'w') as f:
                    json.dump(model_json, f)
                model.save_model(os.path.join(save_folder,f'model_{cv_count}.{model_type}'))
            else:
                model.save_model(save_folder + 'model_{}.json'.format(cv_count))
     # else:
     #     if legacy_save:
     #         model.save_model('model_{}.'.format(cv_count)+model_type)
     #     else:
     #         model.save_model('model_{}.json'.format(cv_count))

    return model


def create_model(model_type='xgb', task_type='regression', model_params=None):
    if model_params is None:
        model_params = get_default_params(model_type=model_type)
    else:
        pass
    if model_type == 'lgb':
        model = LGBMRegressor()
        model.set_params(**model_params)

    if model_type == 'xgb':
        model = XGBRegressor()
        model.set_params(**model_params)

    if task_type == 'classification':
        model = XGBClassifier()
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
