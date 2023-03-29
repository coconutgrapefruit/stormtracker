import math
import datetime as dt
import numpy as np
import xarray as xr


def read_storm(file_name, area=(70, 120, 0, 220), scaler=None):
    ds = xr.open_dataset(file_name)
    ds = ds.reset_coords(['time', 'lat', 'lon'])
    darr = ds[['time', 'lat', 'lon', 'dist2land', 'landfall']]

    stime = darr['time'].data
    slat = darr['lat'].data
    slon = darr['lon'].data

    if scaler is None:
        scaler = np.nanmax(darr['dist2land'].data)
    # scaled
    # dist2land and landfall have useful relationship so the same scaler is used. read IBTrACS docs for details.
    storm = np.transpose(np.array([
        (darr['lat'].data - area[2]) / (area[0] - area[2]),
        (darr['lon'].data - area[1]) / (area[3] - area[1]),
        darr['dist2land'].data / scaler,
        darr['landfall'].data / scaler]
    ), axes=(1, 2, 0))

    # only select modern data due to consistency
    # from 1980
    stime = stime[2787:]
    slat = slat[2787:]
    slon = slon[2787:]
    storm = storm[2787:]

    return [storm, stime, slat, slon], scaler


def transform(storm_data, seq_len: int = 40):
    # storm : s*t*f [lat, lon, dist2lant, landfall]
    # stime, slat, slon only for reference
    storm, stime, slat, slon = storm_data
    x = []
    y = []
    # reference of each sample for retrieving CDS data
    ref = []
    num_storms, num_times, num_features = storm.shape
    for i in range(0, num_storms):
        _nat = np.isnat(stime[i, :])
        # too much data, only select some typhoons to train on for demonstration
        # delete if you have enough time / memory
        if _nat[70] or np.nanmin(slon[i]) < 120 or np.nanmax(slon[i]) > 220 or not 8<=int(str(stime[i,seq_len])[5:7])<=10:
            continue

        j = 0
        while not _nat[j + seq_len]:
            x.append(storm[i, j: j + seq_len - 1, :])
            y.append([storm[i, j + seq_len, 0],
                      storm[i, j + seq_len, 1]])

            ref.append((get_time(stime[i, j + seq_len - 1]),
                        math.floor(slat[i, j + seq_len - 1]),
                        math.floor(slon[i, j + seq_len - 1])))
            j = j + 1
    x = np.array(x)
    y = np.array(y)
    return x, y, ref


def _union(a, b):
    n = max(a[0], b[0])
    w = min(a[1], b[1])
    s = min(a[2], b[2])
    e = max(a[3], b[3])
    return [n, w, s, e]


def _longitude(x):
    if x < 0:
        x = x + 360
    return x


# time: (year, index)
def get_time(time_64):
    _year = int(str(time_64)[0:4])
    _month = int(str(time_64)[5:7])
    _day = int(str(time_64)[8:10])
    _hour = int(str(time_64)[11:13])
    start_year = _year if _year < 2005 else _year - (_year - 2005) % 3
    return int(start_year), int((dt.datetime(year=_year, month=_month, day=_day, hour=_hour)
                                 - dt.datetime(year=start_year, month=1, day=1, hour=0)) / dt.timedelta(hours=3))


def transform_cds(file_name) -> np.ndarray:
    nd = xr.open_dataset(file_name).to_array().data
    nd = np.transpose(nd, (1, 0, 2, 3, 4)).reshape((-1, 9, 71, 101))
    return nd
