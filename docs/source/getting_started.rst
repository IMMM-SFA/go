Getting started
===============

About
-----

The Capacity Expansion Regional Feasibility model (**go**) helps us evaluate the feasibility and structure of future electricity capacity expansion plans by siting power plants in areas that have been deemed the least cost option. We can use **go** to gain an understanding of topics such as: 1) whether or not future projected electricity expansion plans from models such as GCAM are possible to achieve, 2) where and which on-the-ground barriers to siting (e.g., protected areas, cooling water availability) may influence our ability to achieve certain expansions, and 3) how power plant infrastructure build outs and value may evolve into the future when considering locational marginal pricing (LMP) based on the supply and demand of electricity from a grid operations model.

Each grid cell in **go** is given an initial value of suitable (0) or unsuitable (1) based on a collection of suitability criteria gleaned from the literature. **go**'s default suitability layers include both those that are common to all thermal technologies as well as technology-specific suitability criteria. Common suitability layers represent categories such as protected lands, critical habitat areas, and much more. Technology-specific suitability layers are those that satisfy requirements that may not be applicable to all technologies. An example would be minimum mean annual flow requirements for cooling water availability for individual thermal technologies.

Though **go** provides sample data to run the conterminous United States (CONUS), it could be extended to function for any country or set of regions that had the following prerequisite data sources:  a spatial representation of substations or electricity transmission infrastructure, a spatial representation of gas pipeline infrastructure if applicable, any regionally applicable spatial data to construct suitability rasters from, access to hourly zonal LMP, and access to technology-specific information and each technologies accompanying electricity capacity expansion plan per region.  The Global Change Analysis Model (`GCAM <https://github.com/JGCRI/gcam-core>`_) is used to build our expansion plans and establish our technology-specific requirements through the end of the century. We derive our LMP from a grid operations model that also is harmonized with GCAM to provide consistent projections of energy system evolution.  See more about how to generalize **go** for your research `here <user_guide.rst#generalization>`_.

We introduce a metric named Net Locational Cost (NLC) that is used compete power plant technologies for each grid cell based on the least cost option to site. NLC is calculated by subtracting the Net Operating Value (NOV) of a proposed power plant from the cost of its interconnection to the grid to represent the potential deployment value. Both the NOV parameter which incorporates many technology-specific values such as variable operations and maintenance costs, carbon price, heat rate, etc. and the interconnection cost parameter used for both electricity transmission and gas pipelines are configurable per time step.  All equations used in **go** are described in detail in the `documentation <user_guide.rst#fundamental-equations-and-concepts>`_.


Python version support
----------------------

Officially Python 3.9 - 3.11


Installation
------------

.. note::

  **go** requires the use of a solver as accessed by ``pyomo``.  Find information about supported solvers here:  `pyomo solvers <https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers>`_`

**go** can be installed via pip by running the following from a terminal window::

    pip install go

It may be favorable to the user to create a virtual environment for the **go** package to minimize package version conflicts.  See `creating virtual environments <https://docs.python.org/3/library/venv.html>`_ to learn how these function and can be setup.

Installing package data
-----------------------

**go** requires package data to be installed from Zenodo to keep the package lightweight.  After **go** has been installed, run the following from a Python prompt:

.. code-block:: python

    import go

    go.install_package_data()

This will automatically download and install the package data necessary to run the examples in accordance with the version of **go** you are running.


Dependencies
------------

=============   ================
Dependency      Minimum Version
=============   ================
PyYAML          5.4.1
requests        2.25.1
pyomo           6.0.1
=============   ================


Optional dependencies
---------------------

==================    ================
Dependency            Minimum Version
==================    ================
build                 0.5.1
nbsphinx              0.8.6
setuptools            57.0.0
sphinx                4.0.2
sphinx-panels         0.6.0
sphinx-rtd-theme      0.5.2
twine                 3.4.1
pytest                6.2.4
pytest-cov            2.12.1
==================    ================
