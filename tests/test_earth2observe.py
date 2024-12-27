import glob
import os
import shutil
from typing import List

import pytest

from earth2observe.chirps import CHIRPS
from earth2observe.earth2observe import Earth2Observe
from earth2observe.s3 import S3


class TestChirpsBackend:
    @pytest.fixture(scope="module")
    def test_chirps_data_source_instantiate_object(
        self,
        chirps_data_source: str,
        dates: List,
        daily_temporal_resolution: str,
        chirps_variables: List[str],
        lat_bounds: List,
        lon_bounds: List,
        chirps_data_source_output_dir: str,
    ):
        e2o = Earth2Observe(
            data_source=chirps_data_source,
            start=dates[0],
            end=dates[1],
            variables=chirps_variables,
            lat_lim=lat_bounds,
            lon_lim=lon_bounds,
            temporal_resolution=daily_temporal_resolution,
            path=chirps_data_source_output_dir,
        )
        assert isinstance(e2o.DataSources, dict)
        assert isinstance(e2o.datasource, CHIRPS)
        assert e2o.datasource.vars == chirps_variables
        assert isinstance(e2o.datasource.lat_lim, list)
        return e2o

    def test_download_chirps_backend(
        self,
        test_chirps_data_source_instantiate_object: CHIRPS,
        chirps_data_source_output_dir: str,
        number_downloaded_files: int,
    ):
        test_chirps_data_source_instantiate_object.download()
        fname = "P_CHIRPS"
        filelist = glob.glob(
            os.path.join(f"{chirps_data_source_output_dir}", f"{fname}*.tif")
        )
        assert len(filelist) == number_downloaded_files
        # delete the files
        try:
            shutil.rmtree(f"{chirps_data_source_output_dir}")
        except PermissionError:
            print("the downloaded files could not be deleted")


# class TestS3Backend:
#     @pytest.fixture(scope="module")
#     def test_s3_data_source_instantiate_object(
#         self,
#         s3_data_source: str,
#         monthly_dates: List,
#         monthly_temporal_resolution: str,
#         s3_era5_variables: List[str],
#         lat_bounds: List,
#         lon_bounds: List,
#         s3_era5_data_source_output_dir: str,
#     ):
#         e2o = Earth2Observe(
#             data_source=s3_data_source,
#             start=monthly_dates[0],
#             end=monthly_dates[1],
#             variables=s3_era5_variables,
#             lat_lim=lat_bounds,
#             lon_lim=lon_bounds,
#             temporal_resolution=monthly_temporal_resolution,
#             path=s3_era5_data_source_output_dir,
#         )
#         assert isinstance(e2o.DataSources, dict)
#         assert isinstance(e2o.datasource, S3)
#         assert e2o.datasource.vars == s3_era5_variables
#         return e2o
#
#     def test_download_s3_backend(
#         self,
#         test_s3_data_source_instantiate_object: S3,
#         s3_era5_data_source_output_dir: str,
#         number_downloaded_files: int,
#     ):
#         test_s3_data_source_instantiate_object.download()
#         filelist = glob.glob(os.path.join(f"{s3_era5_data_source_output_dir}", f"*.nc"))
#         assert len(filelist) == number_downloaded_files
#         # delete the files
#         try:
#             shutil.rmtree(f"{s3_era5_data_source_output_dir}")
#         except PermissionError:
#             print("the downloaded files could not be deleted")
