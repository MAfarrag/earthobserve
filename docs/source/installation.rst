############
Installation
############


**************
Stable release
**************

- Please install earth2observe in a Virtual environment so that its requirements don't tamper with your system's python
    ``earth2observe`` works with all Python versions

conda
-----
- The easiest way to install ``earth2observe`` is using ``conda`` package manager. ``earth2observe`` is available in
    the `conda-forge`_ channel. To install you can use the following command:

.. code-block:: bash

    conda install -c conda-forge earth2observe

- If this works it will install earth2observe with all dependencies including Python and gdal,
    and you skip the rest of the installation instructions.


Installing Python and gdal dependencies
---------------------------------------

- The main dependencies for earth2observe are an installation of Python 3.9+, and gdal

Installing Python
-----------------

- For Python we recommend using the Anaconda Distribution for Python 3, which is available for download from
    https://www.anaconda.com/download/. The installer gives the option to add ``python`` to your ``PATH`` environment
    variable. We will assume in the instructions below that it is available in the path, such that ``python``,
    ``pip``, and ``conda`` are all available from the command line.

Install as a conda environment
------------------------------

- The easiest and most robust way to install earth2observe is by installing it in a separate conda environment. In
    the root repository directory there is an ``environment.yml`` file. This file lists all dependencies. Either use
    the ``environment.yml`` file from the main branch (please note that the main branch can change rapidly and break
    functionality without warning), or from one of the releases {release}.

- Run this command to start installing all earth2observe dependencies:

.. code-block:: bash

    conda env create -f environment.yml

- This creates a new environment with the name ``earth2observe``. To activate this environment in a session, run:

.. code-block:: bash

    activate earth2observe

- For the installation of earth2observe there are two options (from the Python Package Index (PyPI) or from Github).
    To install a release of earth2observe from the PyPI:

.. code-block:: bash

    pip install earth2observe=={release}


From sources
------------

- The sources for earth2observe can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/Serapieum-of-alex/earth2observe

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/Serapieum-of-alex/earth2observe/tarball/main

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


To install directly from GitHub (from the HEAD of the main branch):

.. code-block:: bash

    pip install git+https://github.com/Serapieum-of-alex/earth2observe.git

or from Github from a specific release:

.. code-block:: bash

    pip install git+https://github.com/Serapieum-of-alex/earth2observe.git@{release}

Now you should be able to start this environment's Python with ``python``, try
``import earth2observe`` to see if the package is installed.


More details on how to work with conda environments can be found here:
https://conda.io/docs/user-guide/tasks/manage-environments.html


If you are planning to make changes and contribute to the development of earth2observe, it is
best to make a git clone of the repository, and do a editable install in the location
of you clone. This will not move a copy to your Python installation directory, but
instead create a link in your Python installation pointing to the folder you installed
it from, such that any changes you make there are directly reflected in your install.

.. code-block:: bash

    git clone https://github.com/Serapieum-of-alex/earth2observe.git
    cd earth2observe
    activate earth2observe
    pip install -e .

Alternatively, if you want to avoid using ``git`` and simply want to test the latest
version from the ``main`` branch, you can replace the first line with downloading
a zip archive from GitHub: https://github.com/Serapieum-of-alex/earth2observe/archive/main.zip
`libraries.io <https://libraries.io/github/Serapieum-of-alex/earth2observe>`_.

Install using pip
-----------------

Besides the recommended conda environment setup described above, you can also install
earth2observe with ``pip``. For the more difficult to install Python dependencies, it is best to
use the conda package manager:

+ ``conda install numpy scipy gdal netcdf4 pyproj``


you can check `libraries.io <https://libraries.io/github/Serapieum-of-alex/earth2observe>`_. to check versions of the libraries


Then install a release {release} of earth2observe (available from release 2018.1) with pip:

+ ``pip install earth2observe=={release}``


Check if the installation is successful
---------------------------------------

To check it the install is successful, go to the examples directory and run the following command:

+ ``python -m earth2observe.*******``

This should run without errors.


.. note::

      This documentation was generated on |date|

      Documentation for the development version:
      https://earth2observe.readthedocs.org/en/latest/

      Documentation for the stable version:
      https://earth2observe.readthedocs.org/en/stable/

References
----------

.. target-notes::
.. _Github repo: https://github.com/Serapieum-of-alex/earth2observe
.. _tarball: https://github.com/Serapieum-of-alex/earth2observe/tarball/main
.. _`conda-forge`: https://conda-forge.org
