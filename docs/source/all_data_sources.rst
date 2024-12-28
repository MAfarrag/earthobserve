*************
Earth2Observe
*************

--------------
Design Concept
--------------
earth2Observe is designed following the Template/Factory design pattern to create an abstract class as a template
for different data sources.

The main objective of the `earth2Observe` is to provide a unified API for all remote sensing data sources, where you
only have to worry about the domain of your data (date range and spatial extent) and the package does everything in
the backend. For some data source like google earth engine and ECMWF you still have to register and create access key.


--------------
Design Concept
--------------

`earth2observe` provides a unified API for the following data sources
- ECMWF
- CHIRPS
- Amazon-S3
- Google Earth Engine (still under development)

Some of the data sources provides the data in servers that needs an authentication key, to create the authentication
key required for each data source check the authentication instruictions .

The API takes few parameters to determine the domain of your data
- The date range you want to download the data within in the form of `start`, `end`, and the `temporal_resolution`
- The spatial extent of your area of interest in the form of latitude and longitude limit `latlim`, `lonlim`
- If the user does not provide the `latlim`, `lonlim` the `Earth2Observe` class will be initialized with the extent
longitude limit [-180, 180], and latitude limit [-90, 90].

.. code-block:: py

    from earth2observe.earth2observe import Earth2Observe
    start = "2009-01-01"
    end = "2009-01-10"
    temporal_resolution = "daily"
    latlim = [4.19, 4.64]
    lonlim = [-75.65, -74.73]


Each data source has different climate variables/datasets, and to know what variables/datasets each data source provides
you can use the `Catalog` class that is with each data source. in this tutorial we will be downloading
`precipitation` from the `CHIRPS` and `Amazon S3` data source and `temperature` from `ECMWF`

- The downloaded data format differs also based on the data source the `CHIRPS` and the `ECMWF` has a `post_download`
    function that is triggered after a the end of each downloaded chunck of data is done, and this `post_download`
    functions triggers some function from the `Pyramids GIS package`_. to convert the netcdf format into geotiff.

- If you like to convert the downloaded data to anyother format or crop the data to a smaller area that what you have
    originally downloaded, reproject, or re-align the rasters to have the same alignment (rows and columns) like other
    raster data you can also find all the required functions in the `Pyramids GIS package`_.

.. _Pyramids GIS package: https://github.com/Serapieum-of-alex/pyramids

.. note::

    in future versions the latlim and lonlim will be deprecated and will be replaced by a geodataframe containing a
    polygon geometry

-----
ECMWF
-----

.. code-block:: py

    source = "ecmwf"
    path = r"examples\data\ecmwf"
    variables = ["precipitation"]

    e2o = Earth2Observe(
        data_source=source,
        start=start,
        end=end,
        variables=variables,
        lat_lim=latlim,
        lon_lim=lonlim,
        temporal_resolution=temporal_resolution,
        path=path,
    )
    e2o.download()

------
CHIRPS
------

.. code-block:: py

    source = "chirps"
    path = r"examples\data\chirps"
    variables = ["precipitation"]
    e2o = Earth2Observe(
        data_source=source,
        start=start,
        end=end,
        variables=variables,
        lat_lim=latlim,
        lon_lim=lonlim,
        temporal_resolution=temporal_resolution,
        path=path,
    )
    e2o.download()

parallel download
-----------------

.. code-block:: py

    path = r"examples\data\chirps-cores"
    e2o = Earth2Observe(
        data_source=source,
        start=start,
        end=end,
        variables=variables,
        lat_lim=latlim,
        lon_lim=lonlim,
        temporal_resolution=temporal_resolution,
        path=path,
    )
    e2o.download(cores=4)

---------
Amazon-S3
---------

.. code-block:: py

    path = r"examples\data\s3-backend"
    source = "amazon-s3"
    variables = ["precipitation"]
    e2o = Earth2Observe(
        data_source=source,
        start=start,
        end=end,
        variables=variables,
        # lat_lim=latlim,
        # lon_lim=lonlim,
        temporal_resolution=temporal_resolution,
        path=path,
    )
    e2o.download()
