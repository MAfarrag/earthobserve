"""Front end module that runs each data source backend."""
from earth2observe.chirps import CHIRPS
from earth2observe.s3 import S3

DEFAULT_LONGITUDE_LIMIT = [-180, 180]
DEFAULT_LATITUDE_LIMIT = [-90, 90]


class Earth2Observe:
    """End user class to call all the data source classes available in earth2observe."""

    DataSources = {"chirps": CHIRPS, "amazon-s3": S3}

    def __init__(
        self,
        data_source: str = "chirps",
        temporal_resolution: str = "daily",
        start: str = None,
        end: str = None,
        path: str = "",
        variables: list = None,
        lat_lim: list = None,
        lon_lim: list = None,
        fmt: str = "%Y-%m-%d",
    ):
        """
        Parameters
        ----------
        data_source: [str]
            data source name. the available data sources are ["ecmwf", "chirps", "amazon-s3"].
        temporal_resolution (str, optional):
            temporal resolution. Defaults to 'daily'.
        start (str, optional):
            start date. Defaults to ''.
        end (str, optional):
            end date. Defaults to ''.
        path (str, optional):
            Path where you want to save the downloaded data. Defaults to ''.
        variables (list, optional):
            Variable name.
        lat_lim (list, optional):
            [ymin, ymax]. Defaults to None.
        lon_lim (list, optional):
            [xmin, xmax]. Defaults to None.
        fmt (str, optional):
            date format. Defaults to "%Y-%m-%d".
        """
        if data_source not in self.DataSources:
            raise ValueError(f"{data_source} not supported")

        if lat_lim is None:
            lat_lim = DEFAULT_LATITUDE_LIMIT
        if lon_lim is None:
            lon_lim = DEFAULT_LONGITUDE_LIMIT

        self.datasource = self.DataSources[data_source](
            start=start,
            end=end,
            variables=variables,
            lat_lim=lat_lim,
            lon_lim=lon_lim,
            temporal_resolution=temporal_resolution,
            path=path,
            fmt=fmt,
        )

    def download(self, progress_bar: bool = True, *args, **kwargs):
        self.datasource.download(progress_bar=progress_bar, *args, **kwargs)
