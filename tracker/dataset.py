import numpy as np
import torch
from torch.utils.data import Dataset
from tracker.datautil import read_storm, transform, transform_cds


class Combined(Dataset):
    def __init__(self,
                 storm: str,
                 cds: str,
                 seq_len: int,
                 scaler=None,
                 scaler_min=np.full(9, np.nan),
                 scaler_max=np.full(9, np.nan),
                 area=(70, 120, 0, 220)
                 ):
        self.storm_file = storm
        self.cds = cds
        self.seq_len = seq_len
        self.scaler = scaler

        self.area = area
        self.scaler_min = scaler_min
        self.scaler_max = scaler_max

        self.x1, self.x2, self.y = self.load()
        self.num_samples = self.y.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

    def get_scalers(self):
        return self.scaler_min, self.scaler_max, self.scaler

    def load(self):
        # IBTrACS data
        storm_data, scaler = read_storm(f'stormdata/{self.storm_file}', area=self.area, scaler=self.scaler)
        self.scaler = scaler
        x2, y, ref = transform(storm_data, self.seq_len)

        # CDS(ERA5) data
        x1 = np.ones((len(ref), 9, 24, 24))
        pre_year = 1980
        graph = transform_cds(f'{self.cds}/1980_.nc')
        i = 0
        for _time, _lat, _lon in ref:
            if _time[0] != pre_year:
                graph = transform_cds(f'{self.cds}/{_time[0]}_.nc')
                pre_year = _time[0]
            i = i+1
            image = graph[_time[1], :, 70-_lat-12:70-_lat+12, _lon-11-120:_lon+13-120]
            print(f'index{i}/{len(ref)}')
            x1[i-1] = image

            self.scaler_min = np.fmin(self.scaler_min, np.amin(image, axis=(1,2)))
            self.scaler_max = np.fmax(self.scaler_max, np.amax(image, axis=(1,2)))

        # scaling
        for i in range(0, 3):
            x1[:, i, :, :] = (x1[:, i, :, :] - self.scaler_min[i])/(self.scaler_max[i] - self.scaler_min[i])
        for i in range(3, 9):
            x1[:, i, :, :] = x1[:, i, :, :]/max(abs(np.amax(self.scaler_max[i])), abs(np.amin(self.scaler_min[i])))

        # to tensor
        x1 = torch.from_numpy(x1.astype(np.float32))
        x2 = torch.from_numpy(x2.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x1, x2, y

