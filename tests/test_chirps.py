import glob
import os
import shutil
from typing import List

import pytest

from earth2observe.chirps import CHIRPS


@pytest.fixture(scope="module")
def test_create_chirps_object(
    dates: List,
    daily_temporal_resolution: str,
    chirps_variables: List[str],
    lat_bounds: List,
    lon_bounds: List,
    chirps_base_dir: str,
):
    coello = CHIRPS(
        start=dates[0],
        end=dates[1],
        lat_lim=lat_bounds,
        lon_lim=lon_bounds,
        variables=chirps_variables,
        temporal_resolution=daily_temporal_resolution,
        path=chirps_base_dir,
    )
    assert coello.api_url == "data.chc.ucsb.edu"
    assert coello.lon_bounds == [-180, 180]
    assert coello.lat_bounds == [-50, 50]
    assert str(coello.time["start_date"].date()) == dates[0]
    assert str(coello.time["end_date"].date()) == dates[1]

    return coello


def test_download(
    test_create_chirps_object: CHIRPS,
    chirps_base_dir: str,
    number_downloaded_files: int,
):
    fname = test_create_chirps_object.clipped_fname
    test_create_chirps_object.download()

    filelist = glob.glob(os.path.join(f"{chirps_base_dir}", f"{fname}*.tif"))
    assert len(filelist) == number_downloaded_files
    # delete the files
    try:
        shutil.rmtree(f"{chirps_base_dir}")
    except PermissionError:
        print("the downloaded files could not be deleted")
