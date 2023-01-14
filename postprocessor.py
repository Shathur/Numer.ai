import os
import numerapi
import pandas as pd
from tqdm import tqdm
from preprocessor import Preprocessor
from predictions import get_predictions_per_era
from utils import get_era_idx
from setup_env_variables import setup

class PostProcessor():
    def __init__(self,preprosessor:Preprocessor,models_path:str,prefix:str):
        self.preprocessor = preprosessor
        self.models_path = models_path
        self.prefix = prefix
        self.predictions_gathered_df = pd.DataFrame()
        self.napi = self.get_napi()
        

    def get_keys(self):
        """get keys from global variables"""
        # setup env variables
        setup()
        # load keys from global environment
        public_id = os.getenv('PUBLIC_ID')
        secret_key = os.getenv('SECRET_KEY')
        return public_id, secret_key

    def get_napi(self):
        """get numerapi instance with specific keys"""
        public_id, secret_key = self.get_keys()
        return numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
    
    def get_predictions_per_era(self,df,model_path):
        """Return predictions per era for our df"""
        return get_predictions_per_era(
            df=df,
            num_models=len(os.listdir(model_path)),
            prefix=self.prefix,
            folder_name=model_path,
            era_idx=get_era_idx(df),
            model_type=self.preprocessor.model_type,
            rank_average=False,
            verbose=False
        ) 

    def add_predictions_per_era(self,df:pd.DataFrame(),model_name:str):
        predictions = self.get_predictions_per_era(df,os.path.join(self.models_path,model_name))
        self.predictions_gathered_df[model_name] = predictions

    def submit_diagnostics(self):
        # get_keys
        public_key,secret_key = self.get_keys()
        # keep our model names as list to iterate on
        model_names = self.predictions_gathered_df.columns.tolist()
        predictions_df = self.predictions_gathered_df['id'].copy().to_frame()
        for name in tqdm(model_names):
            predictions_df['prediction'] = self.predictions_gathered_df[name]
            model_id = self.napi.get_models()[name]
            # Upload predictions
            predictions_df.to_csv('diagnostics.csv', index=False)
            submission_id = self.napi.upload_diagnostics('diagnostics.csv', model_id=model_id)
