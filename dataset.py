import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import normalize_apnea_data, compute_FFT_features

def binary_to_smooth(arr):
    y_smooth = np.zeros(len(arr))
    z_arr = np.array([0])
    arr_pad = np.hstack((np.hstack((z_arr,arr)),z_arr))
    Ends = np.where(((arr_pad[:-1]>arr_pad[1:])[1:]))[0]
    Begins = np.where(((arr_pad[:-1]<arr_pad[1:])[:-1]))[0]
    for k in range(len(Begins)):
        l = Ends[k]+1-Begins[k]
        midd = l//2
        y_smooth[Begins[k]:Ends[k]+1] = np.exp(-((np.arange(l)-midd*np.ones(l)))**2/((midd+pow(10,-5))**2))
    return(y_smooth)


class SleepApneaDataset(torch.utils.data.Dataset):

  def __init__(self, data_df, target_df, p, test=False):

    self.data_df = data_df
    self.target_df = target_df
    self.signal_ids = p.signal_ids
    self.n_signal = len(p.signal_ids)
    self.seq_length = p.seq_length
    self.signal_dim = p.sampling_freq*p.seq_length
    self.sampling_freq = p.sampling_freq
    self.model = p.model
    self.smooth_y = p.smooth_y
    self.test = test

  def __len__(self):
    return len(self.data_df)

  def __getitem__(self, idx):

    sample_index = self.data_df.iloc[idx, 0]
    subject_index = self.data_df.iloc[idx, 1]

    x = np.zeros((self.n_signal, self.signal_dim))
    for i, signal_id in enumerate(self.signal_ids):
      x[i] = self.data_df.iloc[idx, 2+self.signal_dim*signal_id:2+self.signal_dim*(signal_id+1)].values
    x = x.reshape(self.n_signal, self.seq_length, self.sampling_freq)
    x = normalize_apnea_data(x)
    if self.model in ['grouped_conv1d']:
      x = x.squeeze(0)
    elif self.model in ['rnn', 'lstm', 'transformer']:
      x = x.reshape(self.seq_length, self.sampling_freq*self.n_signal)
    if self.test:
      return x, sample_index, subject_index
    else:
      y = self.target_df[self.target_df['ID'] == sample_index].values[0][1:]
      if self.smooth_y:
        y = binary_to_smooth(y)
      return x, y


class EmbeddedDataset(torch.utils.data.Dataset):

  def __init__(self, data_df, target_df, p, test=False):

    self.data_df = data_df
    self.target_df = target_df
    self.signal_ids = p.signal_ids
    self.n_signal = len(p.signal_ids)
    self.signal_dim = p.sampling_freq*p.seq_length
    self.sampling_freq = p.sampling_freq
    self.discrete_transform_type = p.discrete_transform_type
    self.max_order = p.max_order
    self.smooth_y = p.smooth_y
    self.test = test

  def __len__(self):
    return len(self.data_df)

  def __getitem__(self, idx):

    sample_index = self.data_df.iloc[idx, 0]
    subject_index = self.data_df.iloc[idx, 1]
    x = np.zeros((self.n_signal, self.signal_dim))
    for i, signal_id in enumerate(self.signal_ids):
      x[i] = self.data_df.iloc[idx, 2+self.signal_dim*signal_id:2+self.signal_dim*(signal_id+1)].values
    x = x.reshape(self.n_signal, -1, self.sampling_freq)
    if self.discrete_transform_type == 'fft':
      x = compute_FFT_features(x, max_order=self.max_order)
    else:
      raise NotImplementedError('"fft" is the only supported discrete transform atm')
    if self.test:
      return x, sample_index, subject_index
    else:
      y = self.target_df[self.target_df['ID'] == sample_index].values[0][1:]
      if(self.smooth_y):
        y = binary_to_smooth(y)
      return x, y


class SleepApneaDataModule():

    def __init__(self, p):

        self.save_csv = p.save_csv
        self.data_dir = p.data_dir
        self.train_data_file = p.train_data_file
        self.test_data_file = p.test_data_file
        self.val_size = p.val_size
        self.target_file = p.target_file
        self.seed = p.seed
        self.p = p

    def setup(self):

        tqdm.pandas()

        if Path(self.data_dir, 'train.csv').exists() and Path(self.data_dir, 'val.csv').exists():
            print(f'Loading train data from file...')
            train_df = pd.read_csv(Path(self.data_dir, 'train.csv'))
            print(f'...done.')
            print(f'Loading validation data from file...')
            val_df = pd.read_csv(Path(self.data_dir, 'val.csv'))
            print(f'...done.')
        else:
            train_df = pd.DataFrame(np.array(h5py.File(Path(self.data_dir, self.train_data_file), mode='r')['data']))
            train_df, val_df = train_test_split(train_df, test_size=self.val_size, random_state=self.seed)
            if self.save_csv:
                train_df.to_csv(Path(self.data_dir, f'train.csv'), index=False)
                val_df.to_csv(Path(self.data_dir, f'val.csv'), index=False)

        target_df = pd.read_csv(Path(self.data_dir, self.target_file))

        if Path(self.data_dir, f'test.csv').exists():
            print(f'Loading test data from file...')
            test_df = pd.read_csv(Path(self.data_dir, f'test.csv'))
            print(f'...done.')
        else:
            test_df = pd.DataFrame(np.array(h5py.File(Path(self.data_dir, self.test_data_file), mode='r')['data']))
            if self.save_csv:
                test_df.to_csv(Path(self.data_dir, f'test.csv'), index=False)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        self.train_dataset, self.val_dataset, self.test_dataset = (
          SleepApneaDataset(train_df, target_df, self.p),
          SleepApneaDataset(val_df, target_df, self.p),
          SleepApneaDataset(test_df, None, self.p, test=True)
        )


class EmbeddedDataModule():

    def __init__(self, p):

        self.save_csv = p.save_csv
        self.data_dir = p.data_dir
        self.train_data_file = p.train_data_file
        self.test_data_file = p.test_data_file
        self.val_size = p.val_size
        self.target_file = p.target_file
        self.seed = p.seed
        self.p = p

    def setup(self):

        tqdm.pandas()

        if Path(self.data_dir, 'train.csv').exists() and Path(self.data_dir, 'val.csv').exists():
            print(f'Loading train data from file...')
            train_df = pd.read_csv(Path(self.data_dir, 'train.csv'))
            print(f'...done.')
            print(f'Loading validation data from file...')
            val_df = pd.read_csv(Path(self.data_dir, 'val.csv'))
            print(f'...done.')
        else:
            train_df = pd.DataFrame(np.array(h5py.File(Path(self.data_dir, self.train_data_file), mode='r')['data']))
            train_df, val_df = train_test_split(train_df, test_size=self.val_size, random_state=self.seed)
            if self.save_csv:
                train_df.to_csv(Path(self.data_dir, f'train.csv'), index=False)
                val_df.to_csv(Path(self.data_dir, f'val.csv'), index=False)

        target_df = pd.read_csv(Path(self.data_dir, self.target_file))

        if Path(self.data_dir, f'test.csv').exists():
            print(f'Loading test data from file...')
            test_df = pd.read_csv(Path(self.data_dir, f'test.csv'))
            print(f'...done.')
        else:
            test_df = pd.DataFrame(np.array(h5py.File(Path(self.data_dir, self.test_data_file), mode='r')['data']))
            if self.save_csv:
                test_df.to_csv(Path(self.data_dir, f'test.csv'), index=False)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        self.train_dataset, self.val_dataset, self.test_dataset = (
          EmbeddedDataset(train_df, target_df, self.p),
          EmbeddedDataset(val_df, target_df, self.p),
          EmbeddedDataset(test_df, None, self.p, test=True)
        )