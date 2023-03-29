import cdsapi

"""
To download the ERA5 reanalysis data, you need to register a CDS account and agree with the terms & conditions
Follow the steps here https://cds.climate.copernicus.eu/api-how-to

Your request might be denied if the size of the items is too large, try downloading year by year. 
"""


def main():
    d = CDSDownloader()
    for i in range(1980, 2023):
        d.batch_download_year(i)


if __name__ == '__main__':
    main()


class CDSDownloader:
    def __init__(self):
        self.c = cdsapi.Client()
        self.path = 'data'

    def batch_download(self):
        _area = [70, 120, 0, -40]
        self.c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['geopotential', 'u_component_of_wind', 'v_component_of_wind'],
                'pressure_level': ['700', '500', '225'],
                'year': [
                    '1980', '1981', '1982', '1983', '1984', '1985',
                    '1986', '1987', '1988', '1989', '1990', '1991',
                    '1992', '1993', '1994', '1995', '1996', '1997',
                    '1998', '1999', '2000', '2001', '2002', '2003',
                    '2004', '2005', '2006', '2007', '2008', '2009',
                    '2010', '2011', '2012', '2013', '2014', '2015',
                    '2016', '2017', '2018', '2019', '2020', '2021', '2022'
                    ],
                'month': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12'
                    ],
                'day': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24',
                    '25', '26', '27', '28', '29', '30',
                    '31'
                    ],
                'time': ['00:00', '03:00', '06:00', '09:00',
                         '12:00', '15:00', '18:00', '21:00'],
                'format': 'netcdf',  # Supported format: grib and netcdf. Default: grib
                'area': _area,  # North, West, South, East.          Default: global
                'grid': [1.0, 1.0],  # Latitude/longitude grid.           Default: 0.25 x 0.25
            },
            f"{self.path}/new/batched.nc")

    def batch_download_year(self, year):
        _area = [70, 120, 0, -40]
        self.c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['geopotential', 'u_component_of_wind', 'v_component_of_wind'],
                'pressure_level': ['700', '500', '225'],
                'year': year,
                'month': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12'
                    ],
                'day': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24',
                    '25', '26', '27', '28', '29', '30',
                    '31'
                    ],
                'time': ['00:00', '03:00', '06:00', '09:00',
                         '12:00', '15:00', '18:00', '21:00'],
                'format': 'netcdf',  # Supported format: grib and netcdf. Default: grib
                'area': _area,  # North, West, South, East.          Default: global
                'grid': [1.0, 1.0],  # Latitude/longitude grid.           Default: 0.25 x 0.25
            },
            f"{self.path}/new/{year}.nc")



