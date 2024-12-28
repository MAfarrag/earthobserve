import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict


class AbstractDataSource(ABC):
    """Blueprint for all class for different datasource's."""

    def __init__(
        self,
        start: str = None,
        end: str = None,
        variables: list = None,
        temporal_resolution: str = "daily",
        lat_lim: list = None,
        lon_lim: list = None,
        fmt: str = "%Y-%m-%d",
        path: str = "",
    ):
        """

        Parameters
        ----------
        temporal_resolution (str, optional):
            [description]. Defaults to 'daily'.
        start (str, optional):
            [description]. Defaults to ''.
        end (str, optional):
            [description]. Defaults to ''.
        path (str, optional):
            Path where you want to save the downloaded data. Defaults to ''.
        variables (list, optional):
            Variable code: VariablesInfo('day').descriptions.keys(). Defaults to [].
        lat_lim (list, optional):
            [ymin, ymax]. Defaults to None.
        lon_lim (list, optional):
            [xmin, xmax]. Defaults to None.
        fmt (str, optional):
            [description]. Defaults to "%Y-%m-%d".
        """
        self.initialize()
        self.temporal_resolution = temporal_resolution
        self.vars = variables

        self.create_grid(lat_lim, lon_lim)
        self.check_input_dates(start, end, temporal_resolution, fmt)
        self.path = Path(path).absolute()
        # Create the directory
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    @abstractmethod
    def check_input_dates(
        self, start: str, end: str, temporal_resolution: str, fmt: str
    ):
        """Check the validity of input dates."""
        pass

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """Initialize connection with the data source server (for non-ftp servers)"""
        pass

    @abstractmethod
    def create_grid(self, lat_lim: list, lon_lim: list):
        """create a grid from the lat/lon boundaries."""
        pass

    @abstractmethod
    def download(self):
        """Wrapper over all the given variables."""
        # loop over dates if the downloaded rasters/netcdf are for a specific date out of the required
        # list of dates
        pass

    # @abstractmethod
    def download_dataset(self):
        """Download single variable/dataset."""
        # used for non-ftp servers
        pass

    @abstractmethod
    def API(self):
        """send/receive request to the dataset server."""
        pass


class AbstractCatalog(ABC):
    """abstract class for the datasource catalog."""

    def __init__(self):
        self.catalog = self.get_catalog()

    # @abstractmethod
    def get_catalog(self):
        """read the catalog of the datasource from disk or retrieve it from server."""
        pass

    # @abstractmethod
    def get_dataset(self, var_name) -> Dict[str, str]:
        """get the details of a specific variable."""
        return self.catalog.get(var_name)
