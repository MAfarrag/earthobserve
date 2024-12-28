"""Catalog of the Earth2Observe GEE datasource."""
import os
from typing import Dict, List
from dataclasses import dataclass
import yaml
from earth2observe.abstractdatasource import AbstractCatalog
from earth2observe.gee import __path__


@dataclass
class DatasetInfo:
    """Dataset."""

    name: str
    provider: str
    url: str
    spatial_resolution: int
    temporal_resolution: int
    start_date: str
    end_date: str
    bands: List[str]
    bands_description: List[str]
    min: List[int]
    max: List[int]
    date_format: str = "%Y-%m-%d"


class Catalog(AbstractCatalog):
    """GEE Data Catalog."""

    def __init__(self):
        """Initialize the catalog.
        The constructor reads the catalog from disk and assigns it to the `catalog` attribute.

        Examples
        --------
        >>> catalog = Catalog()
        >>> print(catalog)
        ls9_sr:
          name: USGS Landsat 9 Level 2, Collection 2, Tier 1, Surface reflectance
          provider: USGS (https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products)
          url: LANDSAT/LC09/C02/T1_L2
          spatial_resolution: 30
          temporal_resolution: 16
          start_date: '2021-10-31'
          end_date: Now
          bands: [SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, SR_QA_AEROSOL]
          bands_description: ['(ultra blue, coastal aerosol) surface reflectance', blue) surface reflectance,
            (green) surface reflectance, (red) surface reflectance, (near infrared) surface reflectance, (shortwave
              infrared 1) surface reflectance, (shortwave infrared 2) surface reflectance, Aerosol attributes]
          max: [65455, 65455, 65455, 65455, 65455, 65455, 65455, 65455]
          min: [1, 1, 1, 1, 1, 1, 1, 1]
        """
        super().__init__()

    @property
    def datasets(self) -> List[str]:
        """get the list of available datasets.

        Examples
        --------
        >>> catalog = Catalog()
        >>> print(catalog.datasets)
        ['ls9_sr']
        """
        return list(self.catalog.keys())

    def get_dataset(self, var_name) -> Dict[str, str]:
        """get the variable information from the catalog."""
        return DatasetInfo(**super().get_dataset(var_name))

    def get_catalog(self) -> Dict:
        """read the catalog of the datasource from disk or retrieve it from server.


            get_catalog retrieves the dataset catalog

        Returns
        -------
        DataFrame
        """
        with open(os.path.join(__path__[0], "catalog.yaml"), 'r') as file:
            data = yaml.safe_load(file)
        return data

    def __getitem__(self, dataset_id: str) -> DatasetInfo:
        return DatasetInfo(**self.catalog[dataset_id])

    def __repr__(self):
        return yaml.dump(self.catalog, default_flow_style=False, sort_keys=False)

    def __str__(self):
        return yaml.dump(self.catalog, default_flow_style=None, sort_keys=False, width=100)

