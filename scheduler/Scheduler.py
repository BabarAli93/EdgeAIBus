import pandas as pd
import numpy as np
import os
import pickle
from typing import (
    Dict,
    Any
)

import re
from itertools import product
import matplotlib.pyplot as plt
import warnings

import time
import datetime
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates

import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import RMSE
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic, augment_calendar_df
from neuralforecast.utils import AirPassengersDF

from sklearn.metrics import mean_absolute_error, r2_score

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

class Scheduler:
    def __init__(self, config: Dict[str, Any]):

        self.prediction_length = config['prediction_length']
        self.bitbrains_path = config['bitbrains_path']
        self.scheduler_path = config['scheduler_path']
        self.unique_cores = 3 # considering only 2, 4 and 6 core machines in this dataset
        self.patch_np_preds = None

        # TODO assert for training and testing prediction lengths
        self.df = self.dataset_reading()
        predictions_path = os.path.join(self.scheduler_path, 'patchtst_predictions_np.pkl')
        try:
            if os.path.exists(predictions_path):
                # read the predictions with self to give access to sim_edge_env
                with open(predictions_path, "rb") as f:
                    self.patch_np_preds = pickle.load(f)
                
                model_pred_length = int(self.patch_np_preds[0].shape[0]/self.unique_cores)
                assert self.prediction_length == model_pred_length,\
                    f"Pre-trained model prediction length is different from given. Either train a new model or use the prediction length {model_pred_length}."
            else:
                # load the already trained model if available and make predictions
                base_path = os.path.join(self.scheduler_path, 'patch_checkpoints/')
                existing_folders = os.listdir(base_path)
                numbered_folders = [int(folder) for folder in existing_folders if folder.isdigit()]
                df_train, df_test = self.train_test_split_local
                if numbered_folders:
                    # load the exisig model and make predictions
                    model_path = os.path.join(base_path, str(max(numbered_folders)))
                    nf = NeuralForecast.load(model_path)
                    # TODO: make predictions and save in pickle or csv
                    assert self.prediction_length == nf.h, \
                          f"Pre-trained model prediction length is different from given. Either train a new model or use the prediction length {nf.h}."
                    patchtst_preds, patchtst_np = self.patchtst_pred(model=nf, pred_length=6,
                                                                            df_train=df_train, df_test=df_test)
                    # saving the prediction
                    with open(os.path.join(self.scheduler_path, 'patchtst_predictions_np.pkl'), 'wb') as f:
                        pickle.dump(patchtst_np, f)

                    self.patch_np_preds = patchtst_np
                else:
                    nf = self.patch_training(df_train, self.prediction_length)
                    patchtst_preds, patchtst_np = self.patchtst_pred(model=nf, pred_length=self.prediction_length,
                                                                            df_train=df_train, df_test=df_test)
                    # saving only the numpy predictions where each core has continuous predictions of horizon length
                    # followed by next core predictions e.g; 2 core 6 values, 4 core 6 six values and 6 core 6 values
                    with open(os.path.join(self.scheduler_path, 'patchtst_predictions_np.pkl'), 'wb') as f:
                        pickle.dump(patchtst_np, f)
                    self.patch_np_preds = patchtst_np
        except Exception as e:
            raise Exception(f"PatchTST predictions error in the scheduler.py: {e}")

    def patch_training(self, df_train, pred_length):
        """
            pred_length: it is the length of future predictions. It can be any intger starting from 1
        """
        model = PatchTST(h=pred_length, input_size=48, patch_len=16,
                 stride=8, hidden_size=256, linear_hidden_size=256, batch_size=32,
                 encoder_layers=4, n_heads=32, scaler_type='identity', loss=RMSE(), 
                 valid_loss=RMSE(), learning_rate=1e-4, max_steps=100, activation='ReLU',
                 val_check_steps=50)
        
        nf = NeuralForecast(
            models=[model],
            freq='5min'
        )
        nf.fit(df=df_train, val_size=7858, time_col='ds', target_col='y', id_col='unique_id')
        # save the model here. Make predictions and save them too
        base_path = os.path.join(self.scheduler_path, 'patch_checkpoints/')
        save_path = self.create_incremented_folder(base_path)

        # saving the model
        nf.save(path=save_path, overwrite=True)
        # returning the trained model object
        return nf
    
    @property
    def train_test_split_local(self):
        # splitting into 90-10.
        # TODO find a clean way for train test split to make sure it ends with 2,4 and 6 core complete combo
        # built-in train_test_split failed to grouped all cores at the splitting timestep
        df_train, df_test = self.df[:70731], self.df[70731:]
        return df_train, df_test

    def dataset_reading(self):
        months = ['2013-7', '2013-8', '2013-9']
        files = ['383.csv', '392.csv', '386.csv']
        
        dfs = {file: [] for file in files}
        
        for month in months:
            for file in files:
                file_path = os.path.join(self.bitbrains_path, month, file)
                df = pd.read_csv(file_path, sep=';')
                dfs[file].append(self.df_processing(df))
        
        dfs = {file: pd.concat(dfs[file]).bfill() for file in files}

        start_dates = {
            '2013-07-30 23:00:00': pd.to_datetime('2013-07-31 23:00:00'),
            '2013-08-30 23:00:00': pd.to_datetime('2013-08-31 23:00:00')
        }
        
        for start, end in start_dates.items():
            for key in dfs:
                dfs[key] = self.fill_missing(dfs[key], pd.to_datetime(start), end)
        
        merged_df = pd.concat(dfs.values()).sort_index()
        df = merged_df[['CPU cores', 'CPU usage [%]']]
        df = df[df.index > '2013-06-30 23:55:00']
        df.reset_index(inplace=True)
        df.sort_values(by=['index', 'CPU cores'], inplace=True)
        df.rename(columns={"CPU cores": "unique_id", "index": "ds", "CPU usage [%]": "y"}, inplace=True)

        return df

    def df_processing(self, df):
        df.columns = df.columns.str.replace('\t', '')
        df['DateTime'] = df['Timestamp [ms]'].apply(lambda x: datetime.datetime.fromtimestamp(x).replace(second=0, microsecond=0))
        df.set_index('DateTime', inplace=True)
        df = df.drop(columns=['Timestamp [ms]']).resample('5min').ffill()
        return df

    def fill_missing(self, df, start, end):
        previous_day_data = df[(df.index >= start - pd.DateOffset(days=1)) & (df.index <= end - pd.DateOffset(days=1))]
        missing_period_timestamps = pd.date_range(start=start, end=end, freq='5min')
        replicated_data = previous_day_data.copy()
        replicated_data.index = missing_period_timestamps[:len(previous_day_data)]
        df_filled = pd.concat([df, replicated_data]).sort_index()
        return df_filled
        
    def create_incremented_folder(self, base_path):
        # Get all folder names in the base path
        existing_folders = os.listdir(base_path)
        
        # Extract folders that are purely integers
        numbered_folders = [int(folder) for folder in existing_folders if folder.isdigit()]
        
        # Determine the next folder number
        next_number = max(numbered_folders) + 1 if numbered_folders else 1
        
        # Create the new folder path
        new_folder_path = os.path.join(base_path, str(next_number))
        os.makedirs(new_folder_path)

        return new_folder_path
    
    def patchtst_pred(self, model, pred_length, df_train, df_test, iter:int = None):
        """
            This fucntion takes input:
            model: patchtst trained model object
            pred_length: prediction length or horizon used for training
            df_train: training dataset for auto-regressive mode predictions
            df_test: testing set
            iter: number of predictions. Maximum can be calculated from the testing set. If not given, 
                  then goes for maximum length of predictions
        """
        all_preds = []
        all_preds_array = []
        if not iter:
            iter = int(df_test.shape[0] - (self.unique_cores*pred_length))
        inf_time = []

        for i in range(iter):
            s_time = time.time()
            forecasts = model.predict(df_train)
            inf = time.time() - s_time
            inf_time.append(inf)
            all_preds.append(forecasts)
            all_preds_array.append(forecasts['PatchTST'].values)

            next_timestep = forecasts['ds'].min()
            test_values = df_test[df_test['ds'] == next_timestep]
            
            # Append these test values to df_train for the next iteration
            df_train = pd.concat([df_train, test_values]).reset_index(drop=True)
            
            # Remove the used test values from df_test
            df_test = df_test[~df_test.index.isin(test_values.index)].reset_index(drop=True)

        return all_preds, all_preds_array