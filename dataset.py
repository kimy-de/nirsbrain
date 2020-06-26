import numpy as np
import pandas as pd
import warnings;

warnings.filterwarnings('ignore')
import time
from tqdm import tqdm


class DataSet:

    def __init__(self, df, y=None):
        # interpolation
        self.df = df
        self.y = y
        self.df_columns = list(self.df.columns.values)
        self.dst_list = list(self.df.columns[36:].values)
        self.src_list = list(self.df.columns[1:36].values)
        self.df = pd.DataFrame(self.df.to_numpy())
        self.df.loc[:, 36:] = self.df.loc[:, 36:].interpolate(method='cubic', limit_direction='both', axis=1)
        self.df.loc[:, 36:] = self.df.loc[:, 36:].interpolate(method='linear', limit_direction='both', axis=1)
        self.df[self.df < 0] = 0

    def scaling_dst(self, data, weight):

        for col in self.dst_list:
            data[col] = data[col] * (data.rho ** 2) * weight ** ((data.rho // 5) - 2)  # hhb/hbo2:10, ca:200, na:

        return data

    def dst_div(self, data):

        epsilon = 1e-16

        for dst_col, src_col in zip(self.dst_list, self.src_list):
            dst_val = data[dst_col]
            src_val = data[src_col] + epsilon
            delta_ratio = dst_val / src_val
            data[dst_col + '_' + src_col + '_ratio'] = delta_ratio
            data = data.fillna(0)

        return data

    def dst_mul(self, data):

        for dst_col, src_col in zip(self.dst_list, self.src_list):
            dst_val = data[dst_col]
            src_val = data[src_col]
            mul = dst_val - src_val
            data[dst_col + '_' + src_col + '_minus'] = mul

        return data

    def dst_fft(self, data):

        real = data[self.dst_list]
        imag = data[self.dst_list]

        for i in range(len(data)):
            real.iloc[i] = real.iloc[i] - real.iloc[i].mean()
            imag.iloc[i] = imag.iloc[i] - imag.iloc[i].mean()

            real.iloc[i] = np.fft.fft(real.iloc[i], norm='ortho').real
            imag.iloc[i] = np.fft.fft(imag.iloc[i], norm='ortho').imag

        real_part = []
        imag_part = []

        for col in self.dst_list:
            real_part.append(col + '_fft_real')
            imag_part.append(col + '_fft_imag')

        real.columns = real_part
        imag.columns = imag_part

        data = pd.concat([data, real, imag], axis=1)
        data = data.fillna(0)
        return data

    def dst_diff(self, data):

        diff = data[self.dst_list].diff(axis=1)
        diff = diff.interpolate(method='linear', limit_direction='both', axis=1)
        diff.rename(columns=lambda x: 'diff' + x[:3], inplace=True)
        data = pd.concat([data, diff], axis=1)

        return data

    def area(self, x):
        area = 0
        for i in range(len(x) - 1):
            area += (x[i] + x[i + 1]) / 2

        return area * 1e-6

    def integral(self, data):

        dst_val = data[self.dst_list]
        src_val = data[self.src_list]
        data['tmp'] = 0

        for i in range(len(data)):
            data['tmp'].iloc[i] = self.area(src_val.iloc[i]) / self.area(dst_val.iloc[i])

        return data

    def logdiv(self, data):
        for dst_col, src_col in zip(self.dst_list, self.src_list):
            dst_val = data[dst_col]
            src_val = data[src_col]
            epsilon = 1e-16
            minus = .33 * ((np.log(abs(dst_val / (src_val + epsilon))) / data.rho) ** 2)
            data[dst_col + '_' + src_col + '_log'] = minus
            data = data.replace([np.inf, -np.inf], 0)
            data = data.fillna(1)

        return data

    def cleaning(self, data):
        data['temp1'] = np.mean(data[data.rho == 10][self.dst_list], 1) > 2e-9
        idx = data[data['temp1'] == True].index
        data.drop(idx, inplace=True)
        y = self.y.drop(idx, axis=0)

        data['temp2'] = np.mean(data[data.rho == 15][self.dst_list], 1) > 1e-11
        idx = data[data['temp2'] == True].index
        data.drop(idx, inplace=True)
        y = y.drop(idx, axis=0)

        data['temp3'] = np.mean(data[data.rho == 20][self.dst_list], 1) > 1e-13
        idx = data[data['temp3'] == True].index
        data.drop(idx, inplace=True)
        y = y.drop(idx, axis=0)

        data['temp4'] = np.mean(data[data.rho == 25][self.dst_list], 1) > 5e-16
        idx = data[data['temp4'] == True].index
        data.drop(idx, inplace=True)
        y = y.drop(idx, axis=0)

        data.drop(['temp1', 'temp2', 'temp3', 'temp4'], axis=1, inplace=True)
        return data, y

    def comb_box(self, data, weight):

        data = self.scaling_dst(data, weight)
        data = self.dst_fft(data)
        data = self.dst_div(data)
        data = self.logdiv(data)
        data = self.dst_diff(data)

        return data

    def data_proccessing(self, training=True):

        # Default processing
        data = pd.DataFrame(self.df.to_numpy(), columns=self.df_columns)

        if training:
            data, self.y = self.cleaning(data)

        data_hbo = self.comb_box(data, 200)
        data_ca = self.dst_mul( data_hbo)
        data_hhb = self.integral(data_ca)
        data_hhb = data_hhb.drop(self.src_list, axis=1)
        data_na = self.comb_box(data, 80)

        return data_hhb, data_hbo, data_na, data_ca, self.y
