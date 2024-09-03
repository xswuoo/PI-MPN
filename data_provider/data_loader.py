import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='',
                 scale=True, timeenc=0, freq='t', args=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.adj_path = args.adj_path
        self.__read_data__()

    def __read_data__(self):
        # df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        '''
        df_raw.columns: ['date', ...(other features)]
        '''

        od_raw = np.load(os.path.join(self.root_path, self.data_path))
        od_raw = od_raw.reshape((od_raw.shape[0], -1))
        od_raw = pd.DataFrame(od_raw)

        start_date = pd.to_datetime('2023-04-01 00:00:00')
        date_series = pd.date_range(start=start_date, periods=len(od_raw), freq='30T')

        od_raw.insert(0, 'date', date_series)
        df_raw = od_raw
        od_raw = np.zeros((df_raw.shape[0], 1))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        column_sums = df_data.sum()
        observation_range = df_raw.shape[0] * 5
        columns_to_drop = column_sums[column_sums < observation_range].index
        columns_to_observe = column_sums[column_sums >= observation_range].index
        df_data = df_data.drop(columns=columns_to_drop)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler = StandardScaler(mean=train_data.values.mean(), std=train_data.values.std())
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            data_stamp[:,4] = df_stamp['date'].apply(lambda x: ((x - start_date) // pd.Timedelta(minutes=30)) % 48)
        time_ind = (df_stamp['date'].values - df_stamp['date'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = time_ind.reshape(-1, 1)

        mean_values = np.mean(data[border1:border2], axis=0)
        min_values = np.min(data[border1:border2], axis=0)
        max_values = np.max(data[border1:border2], axis=0)
        local_stamp = np.concatenate((mean_values.reshape(-1, 1), min_values.reshape(-1, 1), max_values.reshape(-1, 1)),
                                     axis=1)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.time_in_day = time_in_day
        self.data_stamp = data_stamp
        self.local_stamp = local_stamp
        self.index_x = columns_to_observe.values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_index = self.index_x.reshape(1, -1).astype(np.int64)
        return seq_x, seq_y, seq_x_mark,  seq_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
