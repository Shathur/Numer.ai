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
        meta_model=False,
        integers=False,
    ):
        if integers:
            int_suffix='_int8'
        else:
            int_suffix=''
        if train:
            self.napi.download_dataset(f'v4.1/train{int_suffix}.parquet',os.path.join(self.datapath,f'train{int_suffix}.parquet')) 
        if validation:
            self.napi.download_dataset(f'v4.1/validation{int_suffix}.parquet',os.path.join(self.datapath,f'validation{int_suffix}.parquet')) 
        if live:
            self.napi.download_dataset(f'v4.1/live{int_suffix}.parquet',os.path.join(self.datapath,f'live{int_suffix}.parquet')) 
        if live_example_preds:
            self.napi.download_dataset('v4.1/live_example_preds.parquet',os.path.join(self.datapath,'live_example_preds.parquet')) 
        if validation_example_preds:
            self.napi.download_dataset('v4.1/validation_example_preds.parquet',os.path.join(self.datapath,'validation_example_preds.parquet')) 
        if features:
            self.napi.download_dataset('v4.1/features.json',os.path.join(self.datapath,'features.json')) 
        if meta_model:
            self.napi.download_dataset('v4.1/meta_model.parquet',os.path.join(self.datapath,'meta_model.parquet')) 

    def get_data(self,train=False,validation=False,live=False,merge=False,integers=False):
        if integers:
            int_suffix='_int8'
        else:
            int_suffix=''
        if ((os.path.exists(os.path.join(self.datapath,f'train{int_suffix}.parquet'))) and train):
            self.train_df = pd.read_parquet(os.path.join(self.datapath,f'train{int_suffix}.parquet'))
        if ((os.path.exists(os.path.join(self.datapath,f'validation{int_suffix}.parquet'))) and validation):
            self.validation_df = pd.read_parquet(os.path.join(self.datapath,f'validation{int_suffix}.parquet'))
        if ((os.path.exists(os.path.join(self.datapath,f'live{int_suffix}.parquet'))) and live):
            self.live_df = pd.read_parquet(os.path.join(self.datapath,f'live{int_suffix}.parquet'))
        if merge:
            self.train_df = pd.concat([self.train_df,self.validation_df])

    def get_features(self,feature_group):
        """
        features_json contains: 
            feature_stats
            feature_sets
            targets
        feature_group is one of the feature_sets : 
            small
            medium
            full
            v2_equivalent_features
            v3_equivalent_features
            fncv3_features
        """
        possible_feature_list = ["small","medium","full","v2_equivalent_features","v3_equivalent_features","fncv3_features"]
        assert os.path.exists(f'{self.datapath}/features.json'), 'features_json does not exist, need to download it first'
        assert feature_group in possible_feature_list, f"unavailable feature_set name -- use one of the following {possible_feature_list}"
        
        if feature_group == "full":
            self.feature_cols = [f for f in self.train_df.columns.tolist() if "feature" in f]
        else:
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
        sorted_diffs = per_era_corrs_diff.abs().sort_values(ascending=False)
        worst_n = sorted_diffs.head(num_of_features).index.tolist()
        return sorted_diffs, worst_n
        
    def get_cv(self, n_splits=None):
        """
        gets cv_split_data object. Defaut behaviour is passing the n_splits
        in the class Constructor. We may override it here if we want
        """
        if n_splits==None:
            n_splits = self.n_splits
        self.cv_split_data = cv_split_creator(
            df=self.train_df,
            col='era',
            cv_scheme=TimeSeriesSplitGroupsPurged,
            n_splits=n_splits,
            extra_constructor_params={'embg_grp_num':12}
        )

    def get_test_data(self,num_tour_eras):
        unique_train_eras = self.train_df.columns.tolist()
        unique_test_eras = unique_train_eras[-num_tour_eras:]
        self.test_df = self.train_df[self.train_df.isin(unique_test_eras)]
        self.train_df = self.train_df[~self.train_df.isin(unique_test_eras)]

    def save_custom_era_split(
        self,
        filename="full_data_int8.parquet",
        erasplit=[0,-500,-300,-1],
        save_filenames=[
            "new_train.parquet",
            "new_test.parquet",
            "new_validation.parquet",
        ]
    ):
        """
        loads the dataframe of the provided filename and then splits it into train and
        test data.
        erasplit : [train_start, train_end, test_start, test_end, validation_start, validation_end]
        save_filenames : [train_path, test_path, validation_path]
        """
        new_train_df = pd.read_parquet(os.path.join(self.datapath,filename))
        unique_eras = new_train_df["era"].unique().tolist()
        train_eras = unique_eras[erasplit[0]:erasplit[1]]
        test_eras = unique_eras[erasplit[2]:erasplit[3]]
        validation_eras = unique_eras[erasplit[4]:erasplit[5]]
        new_train_df[new_train_df["era"].isin(train_eras)].to_parquet(os.path.join(self.datapath,save_filenames[0]))
        new_train_df[new_train_df["era"].isin(test_eras)].to_parquet(os.path.join(self.datapath,save_filenames[1]))
        new_train_df[new_train_df["era"].isin(validation_eras)].to_parquet(self.datapath,save_filenames[2])

    def train(self,type_of_model,model_params,fit_params,save_to_drive,save_folder,calculate_metrics,plot_metrics,iteration=0):
        """
        wrapper around cross_validate_train
        iteration is here to choose a target in case we have a list of targets
        """
        if type(self.target) != list:
            target = self.target 
        else:
            target = self.target[iteration]

        cross_validate_train(
            feature_names=self.feature_cols,
            cv_split_data=self.cv_split_data,
            target_name=target,
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
