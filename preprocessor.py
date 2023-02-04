import sys
import os
import numerapi
import pandas as pd
import json
import gc
from cv_splits import cv_split_creator, cross_validate_train, TimeSeriesSplitGroupsPurged


class Preprocessor():
    
    def __init__(self,datapath,n_splits,target):
        self.napi = numerapi.NumerAPI()
        self.datapath = datapath
        self.train_df = pd.DataFrame()
        self.validation_df = pd.DataFrame(),
        self.test_df = pd.DataFrame()
        self.feature_cols = []
        self.target = target
        self.n_splits = n_splits
        self.cv_split_data : []

    def download_data(
        self,
        train=False,
        validation=False,
        live=False,
        live_example_preds=False,
        validation_example_preds=False,
        features=False,
        meta_model=False
    ):
        if train:
            self.napi.download_dataset('v4.1/train.parquet',os.path.join(self.datapath,'train.parquet')) 
        if validation:
            self.napi.download_dataset('v4.1/validation.parquet',os.path.join(self.datapath,'validation.parquet')) 
        if live:
            self.napi.download_dataset('v4.1/live.parquet',os.path.join(self.datapath,'live.parquet')) 
        if live_example_preds:
            self.napi.download_dataset('v4.1/live_example_preds.parquet',os.path.join(self.datapath,'live_example_preds.parquet')) 
        if validation_example_preds:
            self.napi.download_dataset('v4.1/validation_example_preds.parquet',os.path.join(self.datapath,'validation_example_preds.parquet')) 
        if features:
            self.napi.download_dataset('v4.1/features.json',os.path.join(self.datapath,'features.json')) 
        if meta_model:
            self.napi.download_dataset('v4.1/meta_model.parquet',os.path.join(self.datapath,'meta_model.parquet')) 

    def get_data(self,train=False,validation=False,merge=False,clear_memory=False):
        if ((os.path.exists(os.path.join(self.datapath,'train.parquet'))) and train):
            self.train_df = pd.read_parquet(os.path.join(self.datapath,'train.parquet'))
        if ((os.path.exists(os.path.join(self.datapath,'validation.parquet'))) and validation):
            self.validation_df = pd.read_parquet(os.path.join(self.datapath,'validation.parquet'))
        if merge:
            self.train_df = pd.concat([self.train_df,self.validation_df])
        if clear_memory:
            del self.train_df,self.validation_df
            gc.collect()

    def get_features(self,feature_group):
        assert os.path.exists(f'{self.datapath}/features.json'), 'features_json does not exist, need to download it first'
        f = open(f'{self.datapath}/features.json')
        features_json = json.load(f)
        self.feature_cols = features_json['feature_sets'][feature_group]

    def per_era_correlations(self,df,features,era_col,target_col):
        """Get the correlation of each era with the designated target"""
        all_feature_corrs = df.groupby(era_col).apply(
            lambda era: era[features].corrwith(era[target_col])
        )
        return all_feature_corrs

    def get_riskiest_features(self, per_era_corrs, num_of_features):
        """per_era_correlations function is designed to return the input for here"""
        sorted_eras = per_era_corrs.index.sort_values()
        first_half = per_era_corrs[:len(per_era_corrs)//2] 
        second_half = per_era_corrs[len(per_era_corrs)//2:]
        first_half_means = per_era_corrs.loc[first_half.index, :].mean()
        second_half_means = per_era_corrs.loc[second_half.index, :].mean()
        per_era_corrs_diff = first_half_means - second_half_means
        sorted_diffs = per_era_corr_diffs.abs().sort_values(ascending=False)
        worst_n = sorted_diffs.head(num_of_features).index.tolist()
        return sorted_diffs, worst_n
        
    def get_cv(self):
        self.cv_split_data = cv_split_creator(
            df=self.train_df,
            col='era',
            cv_scheme=TimeSeriesSplitGroupsPurged,
            n_splits=self.n_splits,
            extra_constructor_params={'embg_grp_num':12}
        )

    def get_test_data(self,num_tour_eras):
        unique_train_eras = self.train_df.columns.tolist()
        unique_test_eras = unique_train_eras[-num_tour_eras:]
        self.test_df = self.train_df[self.train_df.isin(unique_test_eras)]
        self.train_df = self.train_df[~self.train_df.isin(unique_test_eras)]

    def train(self,type_of_model,model_params,fit_params,save_to_drive,save_folder,calculate_metrics,plot_metrics):
        cross_validate_train(
            feature_names=self.feature_cols,
            cv_split_data=self.cv_split_data,
            target_name=self.target,
            train_df=self.train_df,
            tour_df=self.test_df,
            type_of_model=type_of_model,
            model_params=model_params,
            fit_params=fit_params,
            save_to_drive=save_to_drive,
            save_folder=save_folder,
            calculate_metrics=calculate_metrics,
            plot_metrics=plot_metrics
    )
