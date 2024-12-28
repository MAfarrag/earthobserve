=======
History
=======

0.1.5 (2022-12-07) : add ECMWF data catalog
------------------

* First release on PyPI.

0.1.6 (2022-12-26) :
------------------

* Use environment.yaml and requirements.txt instead of pyproject.toml and replace poetry env byconda env
* lock numpy to 1.23.5

0.1.7 (2022-12-26)
------------------

* fix pypi package names in the requirements.txt file
* fix python version in requirement.txt

0.2.0 (2023-01-15)
------------------

* bump up numpy and pyramids versions
* create an abstract class for datasource and catalog as a bluebrint for all data sources
* test all classes in CI
* use pathlib to deal with paths

0.2.1 (2023-01-25)
------------------

* add amazon S3 data source and catalog for the data available in era5 bucket (ERA5 only tested)
* replace utility functions with the serapeum_utils package.

0.2.2 (2023-01-29)
------------------

* add documentation
* bump up pyramids versions

0.3.0 (******)
------------------

* Replace the setup.py file with a pyproject.toml file
*
