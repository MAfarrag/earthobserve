import glob
import os
import shutil
from typing import List

import pytest

from earth2observe.s3 import S3


@pytest.fixture(scope="module")
def test_create_s3_object(
    monthly_dates: List,
    lat_bounds: List,
    lon_bounds: List,
    s3_era5_base_dir: str,
    s3_era5_variables: List[str],
):
    Coello = S3(
        start=monthly_dates[0],
        end=monthly_dates[1],
        lat_lim=lat_bounds,
        lon_lim=lon_bounds,
        path=s3_era5_base_dir,
        variables=s3_era5_variables,
    )
    assert isinstance(Coello, S3)
    return Coello


def test_download(
    test_create_s3_object: S3,
    s3_era5_base_dir: str,
    number_downloaded_files: int,
):
    test_create_s3_object.download()
    filelist = glob.glob(os.path.join(f"{s3_era5_base_dir}", f"*.nc"))
    assert len(filelist) == number_downloaded_files
    # delete the files
    try:
        shutil.rmtree(f"{s3_era5_base_dir}")
    except PermissionError:
        print("the downloaded files could not be deleted")
