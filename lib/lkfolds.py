import pandas as pd
import random
import numpy as np
from typing import List


class LongitudinalKFolds:
    def __init__(self,
                 df: pd.DataFrame,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 demographics: List[str],
                 random_state: int,
                 waves: List[str]):
        np.random.seed(random_state)
        random.seed(random_state)
        self._df = df
        self._train_df = train_df
        self._test_df = test_df
        self._demographics = demographics
        self._random_state = random_state
        self._waves = waves
        self._num_folds = 12

    @property
    def num_folds(self):
        return self._num_folds

    def _generate_column_subsets(self, clean_columns):
        type_dict = {
            'int_symptoms': 'sum_int_',
            'ext_symptoms': 'sum_ext_',
            'dp_symptoms': 'sum_dp',
            'thickness': '_thickavg',
            'volume': '_vol',
            'age': 'age_mri'}
        model_dict = {'model1': dict.fromkeys(type_dict.keys(), None),
                      'model2': dict.fromkeys(type_dict.keys(), None),
                      'model0': dict.fromkeys(type_dict.keys(), None)}
        for (key, val) in type_dict.items():
            model_dict['model1'][key] = [
                col for col in clean_columns if 
                (val in col or val == col) and 'model1' in col]
            model_dict['model2'][key] = [
                col for col in clean_columns 
                if (val in col or val == col) and 'model2' in col]
            model_dict['model0'][key] = [
                col for col in clean_columns 
                if (val in col or val == col) and (('model1' not in col) and ('model2' not in col)) and col != 'age_mri']
        return model_dict

    def _get_clean_columns(self):
        # This part cleans the columns
        columns = list(self._df)
        age_cols = [column for column in columns if 'age_mri' in column]
        clean_columns = [
            column.replace(
                '_f5', '').replace(
                '_f9', '').replace(
                '_f13', '').replace(
                '_5y', '').replace(
                '_9y', '').replace(
                '_13y', '').replace(
                '_5', '').replace(
                '_9m', '').replace(
                '_14', '') for column in columns
            if column not in self._demographics]

        clean_columns = set(clean_columns).union(set(age_cols))
        return clean_columns

    def _get_wave_dicts(self, clean_columns):
        # This part stacks the age waves
        # We have three waves at which we did an MRI scan: 5, 9, 13.
        # In our normative model, we have to concatenate those together to act like samples are iid, 
        # instead of from the same person. We account for one subject having multiple scans in a training set
        # when we generate the 10 folds
        columns = list(self._df)
        five_dict = {}
        nine_dict = {}
        thirteen_dict = {}
        for column in columns:
            if '5' in column:
                for clean_column in clean_columns:
                    if clean_column == column.replace('_5y', '').replace('_f5', '').replace('_5', ''):
                        five_dict[column] = clean_column
            elif '9' in column:
                for clean_column in clean_columns:
                    if clean_column == column.replace('_9y', '').replace('_f9', '').replace('_9m', ''):
                        nine_dict[column] = clean_column
            elif '13' in column or '14' in column:
                for clean_column in clean_columns:
                    if clean_column == column.replace('_13y', '').replace('_f13', '').replace('_14', ''):
                        thirteen_dict[column] = clean_column
            else:
                print(f'The following column was dropped: {column}')
        return {'F05': five_dict, 'F09': nine_dict, 'F13': thirteen_dict}

    def generate_kfolds(self):
        clean_columns = self._get_clean_columns()
        subset_dict = self._generate_column_subsets(clean_columns)
        wave_dicts = self._get_wave_dicts(clean_columns=clean_columns)
        folds = []
        max_test_size = 0
        for k in range(1, self._num_folds+1):
            self._train_df = self._train_df[self._train_df['trainset_session'].isin(self._waves)]
            self._test_df = self._test_df[self._test_df['testset_session'].isin(self._waves)]
            indices = np.array(
                self._train_df[self._train_df['trainset_number'] == k].index)
            test_indices = np.array(
                self._test_df[self._test_df['testset_number'] == k].index)

            sessions = self._train_df.loc[self._train_df['trainset_number'] == k, 'trainset_session'].values
            test_sessions = self._test_df.loc[self._test_df['testset_number'] == k, 'testset_session'].values
            train_indices = []
            valid_indices = []
            train_df_waves = []
            valid_df_waves = []
            test_df_waves = []
            # For each wave we create a training valid and test split
            for wave in self._waves:
                # Calculate the total number of subjects in a wave for the training set
                total = (sessions == wave).sum()
                # Take five percent of that number for the validation set
                five_percent = int(total * 0.05)
                # Permutate the index for each subject in the wave for the training set
                perm = np.random.permutation(np.arange(total))
                temp_indices = indices[sessions == wave].copy()
                # Take five percent random subjects for the validation set
                valid_indices = temp_indices[perm[:five_percent]].copy()
                train_indices = temp_indices[perm[five_percent:]].copy()

                # Put the training subjects for this wave in a dataframe
                train_df_wave = self._df.loc[train_indices, wave_dicts[wave].keys()].copy()
                train_df_wave.columns = wave_dicts[wave].values()
                train_df_wave.index = [f'{ix}{wave}' for ix in train_df_wave.index.to_list()]

                # Put the training subjects for this wave in a dataframe
                valid_df_wave = self._df.loc[valid_indices, wave_dicts[wave].keys()].copy()
                valid_df_wave.columns = wave_dicts[wave].values()
                valid_df_wave.index = [f'{ix}{wave}' for ix in valid_df_wave.index.to_list()]

                # Put the test subjects for this wave in a dataframe
                test_indices_wave = test_indices[test_sessions == wave]
                test_df_wave = self._df.loc[test_indices_wave, wave_dicts[wave].keys()].copy()
                test_df_wave.columns = wave_dicts[wave].values()
                test_df_wave.index = [f'{ix}{wave}' for ix in test_df_wave.index.to_list()]

                # Append the training, validation, test sets to a list
                train_df_waves.append(train_df_wave)
                valid_df_waves.append(valid_df_wave)
                test_df_waves.append(test_df_wave)
            # For each fold, stack the training dataframes for each wave together
            # for the training, validation, and test set
            fold = {
                'train': pd.concat(train_df_waves, axis=0),
                'valid': pd.concat(valid_df_waves, axis=0),
                'test': pd.concat(test_df_waves, axis=0)
            }
            if fold['test'].shape[0] > max_test_size:
                max_test_size = fold['test'].shape[0]
            folds.append(fold)
        return folds, subset_dict, max_test_size