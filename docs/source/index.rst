
Current release info
====================

.. image:: https://readthedocs.org/projects/earth2observe/badge/?version=latest
    :target: https://earth2observe.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/earth2observe
  :target: https://pypi.org/project/earth2observe/


.. image:: https://img.shields.io/conda/v/conda-forge/earth2observe?label=conda-forge
    :target: https://pypi.org/project/earth2observe/


.. image:: https://img.shields.io/pypi/pyversions/earth2observe
    :target: https://pypi.org/project/earth2observe/


.. image:: https://img.shields.io/github/forks/mafarrag/earth2observe?style=social
    :alt: GitHub forks


.. image:: https://anaconda.org/conda-forge/earth2observe/badges/downloads.svg
    :target: https://anaconda.org/conda-forge/earth2observe


.. image:: https://img.shields.io/conda/vn/conda-forge/earth2observe
    :alt: Conda (channel only)
    :target: https://pypi.org/project/earth2observe/0.1.0/


.. image:: https://img.shields.io/gitter/room/mafarrag/earth2observe
  :alt: Gitter


.. image:: https://static.pepy.tech/personalized-badge/earth2observe?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
  :target: https://pypi.org/project/earth2observe/





.. image:: https://anaconda.org/conda-forge/earth2observe/badges/platforms.svg
  :target: https://anaconda.org/conda-forge/earth2observe


.. image:: https://static.pepy.tech/personalized-badge/earth2observe?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
  :target: https://pepy.tech/project/earth2observe



.. image:: https://static.pepy.tech/personalized-badge/earth2observe?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
  :target: https://pepy.tech/project/earth2observe


.. image:: https://static.pepy.tech/personalized-badge/earth2observe?period=week&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
  :target: https://pepy.tech/project/earth2observe




earth2observe - Remote sensing package
=======================================

**earth2observe** is a Python package providing unified API for several remote sensing data sources


Main Features
-------------
  - ERA Interim Download from ECMWF
  - CHIRPS Rainfall data Download
  - ERA5 from Amason S3 data source

.. digraph:: Linking

    earth2observe -> ECMWF;
    earth2observe -> CHIRPS;
    earth2observe -> Amazon-S3;
    earth2observe -> Google-Earth-Engine;
    dpi=200;

.. toctree::
   :hidden:
   :maxdepth: 1

   Installation <installation.rst>
   Authentication <authentication.rst>
   Data Catalog <catalog.rst>
   Data Sources <all_data_sources.rst>
   Data Post processing <post_processing.rst>
