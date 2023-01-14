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
    
    def get_predictions_per_era(self,df,model_path,model_type):
        """Return predictions per era for our df"""
        return get_predictions_per_era(
            df=df[self.preprocessor.feature_cols],
            num_models=len(os.listdir(model_path)),
            prefix=self.prefix,
            folder_name=model_path,
            era_idx=get_era_idx(df),
            model_type=model_type,
            rank_average=False,
            verbose=False
        ) 

    def add_predictions_per_era(self,df:pd.DataFrame(),model_name:str,model_type:str):
        """predict and add prediction to dataframe"""
        predictions = self.get_predictions_per_era(df,os.path.join(self.models_path,model_name),model_type)

    def get_id_column(self, df):
        """return index id as a column"""
        if df.index.name == 'id':
            id_series = df.reset_index()['id']
        else:
            raise ValueError("df should be a DataFrame with an index named 'id'")
        return id_series

    def submit_diagnostics(self, df):
        # get_keys
        public_key,secret_key = self.get_keys()
        # keep our model names as list to iterate on
        model_names = self.predictions_gathered_df.columns.tolist()
        # keep id as a column
        predictions_df = self.get_id_column(df).to_frame()
        for name in tqdm(model_names):
            predictions_df['prediction'] = self.predictions_gathered_df[name]
            model_id = self.napi.get_models()[name]
            # Upload predictions
            predictions_df.to_csv('diagnostics.csv', index=False)
            submission_id = self.napi.upload_diagnostics('diagnostics.csv', model_id=model_id)
