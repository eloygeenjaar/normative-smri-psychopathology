import pandas as pd
from pathlib import Path

from lib.lkfolds import LongitudinalKFolds

class GenerationR:

    def __init__(self,
                 df_path: Path,
                 train_df: Path,
                 test_df: Path,
                 random_state: int,
                 use_last_wave: bool = False):
        self._random_state = random_state
        self._use_last_wave = use_last_wave
        self._df = pd.read_csv(df_path, index_col=0)
        self._train_df = pd.read_csv(train_df, index_col=0)
        self._test_df = pd.read_csv(test_df, index_col=0)
        self._train_df['trainset_number'] = self._train_df['trainset_number'].astype(int)
        self._test_df['testste_number'] = self._test_df['testset_number'].astype(int)

        self._demographics = [
            'GENDER', 'HC12', 'HC12_9', 'H12_13', 'maternal_education',
            'paternal_education', 'household_income', 'child_nationalorigin',
            'age_cbcl_5y', 'age_cbcl_9y', 'age_cbcl_14y'
        ]
        self._waves = [
            'F05', 'F09',
        ]
        self._min_age = self._df['age_mri_5y'].min()
        if use_last_wave:
            self._waves += ['F13']
            self._max_age = self._df['age_mri_13y'].max()
        else:
            self._max_age = self._df['age_mri_9y'].max()
            

        self._kfold = LongitudinalKFolds(
            df=self._df,
            train_df=self._train_df,
            test_df=self._test_df,
            random_state=self._random_state,
            demographics=self._demographics,
            waves=self._waves)
        self._folds, self._model_dict, self._max_test_size \
            = self._kfold.generate_kfolds()

    @property
    def folds(self):
        return self._folds

    @property
    def model_dict(self):
        return self._model_dict

    @property
    def max_test_size(self):
        return self._max_test_size
    
    @property
    def min_max(self):
        return (self._min_age, self._max_age)
