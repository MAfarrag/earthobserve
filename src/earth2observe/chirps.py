import datetime as dt
import os
from ftplib import FTP

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyramids.dataset import Dataset
from pyramids._io import extract_from_gz
from serapeum_utils.utils import print_progress_bar

from earth2observe.abstractdatasource import AbstractCatalog, AbstractDataSource


class CHIRPS(AbstractDataSource):
    """CHIRPS."""

    api_url: str = "data.chc.ucsb.edu"
    start_date: str = "1981-01-01"
    end_date: str = "Now"
    temporal_resolutions = ["daily", "monthly"]
    lat_bounds = [-50, 50]
    lon_bounds = [-180, 180]
    globe_fname = "chirps-v2.0"
    clipped_fname = "P_CHIRPS.v2.0"

    def __init__(
        self,
        temporal_resolution: str = "daily",
        start: str = None,
        end: str = None,
        path: str = "",
        variables: list = None,
        lat_lim: list = None,
        lon_lim: list = None,
        fmt: str = "%Y-%m-%d",
    ):
        """CHIRPS.

        Parameters
        ----------
        temporal_resolution (str, optional):
            'daily' or 'monthly'. Defaults to 'daily'.
        start (str, optional):
            [description]. Defaults to ''.
        end (str, optional):
            [description]. Defaults to ''.
        path (str, optional):
            Path where you want to save the downloaded data. Defaults to ''.
        variables (list, optional):
            Variable code: VariablesInfo('day').descriptions.keys(). Defaults to [].
        lat_lim (list, optional):
            [ymin, ymax] (values must be between -50 and 50). Defaults to [].
        lon_lim (list, optional):
            [xmin, xmax] (values must be between -180 and 180). Defaults to [].
        fmt (str, optional):
            [description]. Defaults to "%Y-%m-%d".
        """
        super().__init__(
            start=start,
            end=end,
            variables=variables,
            temporal_resolution=temporal_resolution,
            lat_lim=lat_lim,
            lon_lim=lon_lim,
            fmt=fmt,
            path=path,
        )

    def check_input_dates(
            self, start_date: str, end_data: str, temporal_resolution: str, fmt: str
    ):
        """check the validity of input dates.

        Parameters
        ----------
        temporal_resolution: (str, optional)
            [description]. Defaults to 'daily'.
        start_date: (str, optional)
            [description]. Defaults to ''.
        end_data: (str, optional)
            [description]. Defaults to ''.
        fmt: (str, optional)
            [description]. Defaults to "%Y-%m-%d".
        """
        # check temporal_resolution variables
        if start_date is None:
            start = pd.Timestamp(self.start_date)
        else:
            start = dt.datetime.strptime(start_date, fmt)

        if end_data is None:
            end = pd.Timestamp(self.end_date)
        else:
            end = dt.datetime.strptime(end_data, fmt)

        # Define timestep for the timedates
        if temporal_resolution.lower() == "daily":
            time_freq = "D"
        elif temporal_resolution.lower() == "monthly":
            time_freq = "MS"
        else:
            raise KeyError("The input temporal_resolution interval is not supported")

        dates = pd.date_range(start, end, freq=time_freq)
        return {"start_date": start, "end_date": end, "time_freq": time_freq, "dates": dates}

    def initialize(self)-> FTP:
        """Initialize FTP server."""
        try:
            ftp = FTP(CHIRPS.api_url)
            ftp.login()
        except Exception:
            raise AuthenticationError("Could not connect to the server")

        return ftp

    def create_grid(self, lat_lim: list, lon_lim: list):
        """Create_grid.

            create grid from the lat/lon boundaries

        Parameters
        ----------
        lat_lim: []
            latitude boundaries
        lon_lim: []
            longitude boundaries
        """
        lat_lim_calc = []
        lon_lim_calc = []
        # Check space variables
        # -50 , 50
        if lat_lim[0] < self.lat_bounds[0] or lat_lim[1] > self.lat_bounds[1]:
            print(
                "Latitude above 50N or below 50S is not possible."
                " Value set to maximum"
            )
            lat_lim_calc[0] = np.max(lat_lim[0], self.lat_bounds[0])
            lat_lim_calc[1] = np.min(lon_lim[1], self.lat_bounds[1])
        # -180, 180
        if lon_lim[0] < self.lon_bounds[0] or lon_lim[1] > self.lon_bounds[1]:
            print(
                "Longitude must be between 180E and 180W. Now value is set to maximum"
            )
            lon_lim_calc[0] = np.max(lat_lim[0], self.lon_bounds[0])
            lon_lim_calc[1] = np.min(lon_lim[1], self.lon_bounds[1])
        else:
            lat_lim_calc = lat_lim
            lon_lim_calc = lon_lim

        # Define IDs
        y_id = 2000 - np.int16(
            np.array(
                [np.ceil((lat_lim[1] + 50) * 20), np.floor((lat_lim[0] + 50) * 20)]
            )
        )
        x_id = np.int16(
            np.array(
                [np.floor((lon_lim[0] + 180) * 20), np.ceil((lon_lim[1] + 180) * 20)]
            )
        )

        return {"x_id": x_id, "y_id": y_id, "lat_lim": lat_lim_calc, "lon_lim": lon_lim_calc}


    def download(self, progress_bar: bool = True, cores=None, *args, **kwargs):
        """Download.

            Download CHIRPS data

        Parameters
        ----------
        progress_bar: bool, optional, The default is True.
            will print a progress bar.
        cores : int, optional, default is None.
            The number of cores used to run the routine. It can be 'False'
                 to avoid using parallel computing routines.

        Returns
        -------
        results : TYPE
            DESCRIPTION.
        """
        # Pass variables to parallel function and run
        args = [
            self.root_dir,
            self.temporal_resolution,
            self.space["x_id"],
            self.space["y_id"],
            self.space["lon_lim"],
            self.space["lat_lim"],
        ]

        if not cores:
            # Create Wait bar
            if progress_bar:
                total_amount = len(self.time["dates"])
                amount = 0
                print_progress_bar(
                    amount,
                    total_amount,
                    prefix="Progress:",
                    suffix="Complete",
                    length=50,
                )

            for date in self.time["dates"]:
                self.api(date, args)

                if progress_bar:
                    amount += 1
                    print_progress_bar(
                        amount,
                        total_amount,
                        prefix="Progress:",
                        suffix="Complete",
                        length=50,
                    )
            results = True
        else:
            results = Parallel(n_jobs=cores)(
                delayed(self.api)(date, args) for date in self.dates
            )
        return results

    def api(self, date, args):
        """form the request url abd trigger the request.

        Parameters
        ----------
        date:

        args: [list]
        """
        [path, temp_resolution, x_id, y_id, lon_lim, lat_lim] = args

        # Define an FTP path to directory
        if temp_resolution.lower() == "daily":
            path_ftp = f"pub/org/chg/products/CHIRPS-2.0/global_daily/tifs/p05/{date.strftime('%Y')}/"
        elif temp_resolution == "monthly":
            path_ftp = "pub/org/chg/products/CHIRPS-2.0/global_monthly/tifs/"
        else:
            raise KeyError("The input temporal_resolution interval is not supported")

        if temp_resolution.lower() == "daily":
            filename = f"{self.globe_fname}.{date.strftime('%Y')}.{date.strftime('%m')}.{date.strftime('%d')}.tif.gz"
            out_file_name = os.path.join(
                path,
                f"{self.globe_fname}.{date.strftime('%Y')}.{date.strftime('%m')}.{date.strftime('%d')}.tif",
            )
            dir_file_end = os.path.join(
                path,
                f"{self.clipped_fname}_mm-day-1_daily_{date.strftime('%Y')}.{date.strftime('%m')}.{date.strftime('%d')}.tif",
            )
        elif temp_resolution == "monthly":
            filename = (
                f"{self.globe_fname}.{date.strftime('%Y')}.{date.strftime('%m')}.tif.gz"
            )
            out_file_name = os.path.join(
                path,
                f"{self.globe_fname}.{date.strftime('%Y')}.{date.strftime('%m')}.tif",
            )
            dir_file_end = os.path.join(
                path,
                f"{self.clipped_fname}_mm-month-1_monthly_{date.strftime('%Y')}.{date.strftime('%m')}.{date.strftime('%d')}.tif",
            )
        else:
            raise KeyError("The input temporal_resolution interval is not supported")

        self.send_request(path_ftp, path, filename)
        self.post_download(
            path, filename, lon_lim, lat_lim, x_id, y_id, out_file_name, dir_file_end
        )

    def send_request(self, ftp_path: str, path: str, filename: str):
        """send the request to the server.

        RetrieveData method retrieves CHIRPS data for a given date from the
        https://data.chc.ucsb.edu/

        Parameters
        ----------
        ftp_path: [str]
            path for the raster in the ftp server.
        filename
        path


        Raises
        ------
        KeyError
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.
        """
        server = self.initialize()
        # find the document name in this directory
        server.cwd(ftp_path)
        listing = []

        # read all the file names in the directory
        server.retrlines("LIST", listing.append)

        local_filename = os.path.join(path, filename)
        lf = open(local_filename, "wb")
        server.retrbinary("RETR " + filename, lf.write, 8192)
        lf.close()

    def post_download(
        self,
        path,
        filename,
        lon_lim,
        lat_lim,
        x_id,
        y_id,
        out_file_name,
        dir_file_end,
    ):
        """clip the downloaded data to the extent we want.

        Parameters
        ----------
        path: [str]
            directory where files will be saved
        filename: [str]
            file name
        lon_lim: [list]
        lat_lim: [list]
        x_id: [list]
        y_id: [list]
        out_file_name: [str]
        dir_file_end: [str]
        """
        try:
            # unzip the file
            zip_filename = os.path.join(path, filename)
            extract_from_gz(zip_filename, out_file_name, delete=True)

            # open tiff file
            dataset = Dataset.read_file(out_file_name)

            data = dataset.read_array()
            no_data_value = dataset.no_data_value[0]

            # clip dataset to the given extent
            data = data[y_id[0]: y_id[1], x_id[0]: x_id[1]]
            # replace -ve values with -9999
            data[data < 0] = -9999

            # save dataset as a geotiff file
            geo = [lon_lim[0], 0.05, 0, lat_lim[1], 0, -0.05]

            new_dataset = Dataset.create_from_array(data, geo=geo, epsg=dataset.epsg, no_data_value=no_data_value)
            new_dataset.to_file(dir_file_end)

            # delete old tif file
            os.remove(out_file_name)

        except PermissionError:
            print(
                "The file covering the whole world could not be deleted please delete it after the download ends"
            )
        return True


class Catalog(AbstractCatalog):
    """CHIRPS data catalog."""

    def __init__(self):
        super().__init__()

    def get_catalog(self):
        """return the catalog."""
        return {
            "Precipitation": {
                "descriptions": "rainfall [mm/temporal_resolution]",
                "units": "mm/temporal_resolution",
                "temporal resolution": ["daily", "monthly"],
                "file name": "rainfall",
                "var_name": "R",
            }
        }

    def get_dataset(self, var_name):
        """get the details of a specific variable."""
        return super().get_dataset(var_name)


class AuthenticationError(Exception):
    """Failed to establish connection with ECMWF server."""

    pass
