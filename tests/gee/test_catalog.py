from typing import List
from earth2observe.gee.catalog import Catalog, DatasetInfo


def test_catalog(catalog_columns: List[str]):
    catalog = Catalog()
    assert hasattr(catalog, "catalog")

    assert isinstance(catalog.datasets, list)
    dataset = catalog["ls9_sr"]
    assert isinstance(dataset, DatasetInfo)
    dataset = catalog.get_dataset("ls9_sr")
    assert all(key in dataset.__dict__.keys() for key in ["name", "provider", "url", "spatial_resolution",
                                                          "temporal_resolution", "start_date", "end_date", "bands",
                                                          "bands_description", "min", "max"])
    assert isinstance(catalog.__str__(), str)
    assert isinstance(catalog.__repr__(), str)
