************
Data Catalog
************

Each data source provide some datasets/climate variables, and the catalog class is the only way
for you to know what data are available at a certain date at a specific location.

The data catalog is a dictionary with the available datasets as keys and the attributes that
describe each dataset are stored in an another dictionary.


------
CHIRPS
------

.. code-block:: py

    from earth2observe.chirps import Catalog
    print(chirps_catalog.catalog)

    >>> {
    >>> 'Precipitation':
    >>>         {
    >>>             'descriptions': 'rainfall [mm/temporal_resolution]',
    >>>             'units': 'mm/temporal_resolution',
    >>>             'temporal resolution': ['daily', 'monthly'],
    >>>             'file name': 'rainfall',
    >>>             'var_name': 'R'
    >>>         }
    >>> }


-----
ECMWF
-----

.. code-block:: py

    from earth2observe.ecmwf import Catalog
    catalog = Catalog()

    >>> {
    >>>     'version': 1,
    >>>     'datasets': [
    >>>                     'cams_gfas', 'cams_nrealtime', 'cera20c', 'cera_sat', 'era15', 'era20c', 'era20cm', 'era20cmv0',
    >>>                     'era40', 'geff_reanalysis', 'icoads', 'interim', 'interim_land', 'ispd', 'macc', 'macc_nrealtime',
    >>>                     's2s', 'tigge', 'uerra', 'yopp', 'yotc'
    >>>                ],
    >>>     'variables': [
    >>>                     'T', '2T', 'SRO', 'SSRO', 'WIND', '10SI', 'SP', 'Q', 'SSR', 'R', 'E', 'SUND', 'RO', 'TP', '10U',
    >>>                     '10V', '2D', 'SR', 'AL', 'HCC'
    >>>                 ],
    >>>     'T': {
    >>>             'descriptions': 'Temperature [K]',
    >>>             'units': 'C',
    >>>             'types': 'state',
    >>>             'temporal resolution': ['six hours', 'daily', 'monthly'],
    >>>             'file name': 'Tair2m',
    >>>             'download type': 3,
    >>>             'number_para': 130,
    >>>             'var_name': 't',
    >>>         },
    >>> .....
    >>> .....
    >>> }

- To get the attributes for a specific variable for example the Evaporation `E`

.. code-block:: py

    var = "E"
    catalog.get_dataset(var)

    >>> {
    >>>     'descriptions': 'Evaporation [m of water]',
    >>>     'units': 'mm',
    >>>     'types': 'flux',
    >>>     'temporal resolution': ['six hours', 'daily', 'monthly'],
    >>>     'file name': 'Evaporation',
    >>>     'download type': 2,
    >>>     'number_para': 182,
    >>>     'var_name': 'e'
    >>> }


---------
Amazon-S3
---------
- for Amazon S3 the data depends on the aws bucket, so the catalog object has to initialize a connection to the
    bucket and check the data inside the bucket

.. code-block:: py

    from earth2observe.s3 import Catalog
    s3_catalog = Catalog()
    print(s3_catalog.catalog)

    >>> {
    >>>     'precipitation': {
    >>>                         'descriptions': 'rainfall [mm/temporal_resolution]',
    >>>                         'units': 'mm/temporal_resolution',
    >>>                         'temporal resolution': ['daily', 'monthly'],
    >>>                         'file name': 'rainfall',
    >>>                         'var_name': 'R',
    >>>                         'bucket_name': 'precipitation_amount_1hour_Accumulation'
    >>>                     }
    >>> }

- As you can see the attribute descibes the same climate variable like precipitation differs from one data source to
    another

- To get the attributes for a specific climate variable.

.. code-block:: py

    s3_catalog.get_dataset("precipitation")

    >>> {
    >>>     'descriptions': 'rainfall [mm/temporal_resolution]',
    >>>     'units': 'mm/temporal_resolution',
    >>>     'temporal resolution': ['daily', 'monthly'],
    >>>     'file name': 'rainfall',
    >>>     'var_name': 'R',
    >>>     'bucket_name': 'precipitation_amount_1hour_Accumulation'
    >>> }

- To get the time span of the precipitation data.

.. code-block:: py

    years = s3_catalog.get_available_years()
    print(years)
    >>> [
    >>>     '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992',
    >>>     '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
    >>>     '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020',
    >>>     '2021', '2022', 'QA', 'zarr'
    >>> ]


.. note::

    The catalog is still in the development phase, ideally the catalog will be json file containing all the available
    data provided by each data source, at the moment the Amazon S3 data source catalog contains only the
