# AI Increases Global Access to Reliable Flood Forecasts

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository allows you to recreate the figures and statistics from the following paper:

[Nearing, Grey, et al. "AI Increases Global Access to Reliable Flood Forecasts." arXiv preprint arXiv:2307.16104 (2023)](<https://arxiv.org/abs/2307.16104>)


## Table of Contents
- [Overview](#overview)
- [License](#license)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Documentation](#documentation)
- [Issues](https://github.com/googlestaging/global_streamflow_model_paper/issues)

-----

## Overview

The code in this repository is structured so that all analysis can be done with python notebooks in the `notebooks` directory. The expected runtime is approxiamtely 1 day for the full analysis, and approximately 5 minutes for repdroducing figures. Also included in the `notebooks` directory is a subdirectory called 'backend', which contains much of the active (mostly functional) code used by the analysis notebooks. The user should only touch the source code in this directory to change their local working directory, as described in the detailed instructions below. 

For the most part, the analysis notebooks must be run sequentially. There are two parts to this process. 

1) The first part includes running all notebooks with names that begin with an integer, in order. These notebooks generally prepare and pre-process data and calculate the necessary evaluation metrics over simulated hydrographs. These notebooks only need to be run once, and they will store results locally. Note that these notebooks are not heavily optimized, and may require significant local memory.

2) The second part includes running all notebooks with names that begin with the word "figure". Each of these notebooks creates one particular figure from the paper referenced above. These figures require data that are calculated and stored (locally) by the preprocessing notebooks. 

Finally, within the `backend` source directory is another source directory called `return_period_calculator`. This subdirectory contains python code for implementing standard methods for fitting return period distributions and estimating return periods. These calculations are based loosely on guidelines outlined in the USGS Bulletin 17c, with some differences related to the statistical tests used for identifying outliers. 

## License
This repository is licensed under an Apache 2.0 open source license. Please see the [LICENSE](https://github.com/googlestaging/global_streamflow_model_paper/blob/main/LICENSE) file in the root directory for the full license. 

This is not an official Google product.

## System Requirements

This repository should run on any computer and operating system that supports Python version 3. It has been tested on the Debian GNU/Linux 11 (bullseye) operating system.

## Installation

No software installation is required beyond Python v3 and Python libraries contained in the environment file. This repository is based on Python notebooks and can be run directly from a local clone:

`git clone https://github.com/googlestaging/global_streamflow_model_paper.git`

An [environment file](https://github.com/googlestaging/global_streamflow_model_paper/blob/main/environment.yml) is included for installing the necessary Python dependencies. If you are using Anaconda (miniconda, etc.) you may create this invironment with the following command from inside the `global_streamflow_model_paper` directory that results from cloning this repository:

`conda env create -f environment.yml`

## Documentation
Detailed Steps to Recreate Results Reported in the Paper

### Step 0: Download Raw Data

#### Download Model Simulations from Zenodo Repository
You will need to download and unzip/untar the `global_modeling_data.tgz` tarball from the Zenodo repository listed in the Code and Data Availability section of the paper referenced at the top of this README document. The DOI for the zenodo repository is: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8139380.svg)](https://doi.org/10.5281/zenodo.8139380)

Your working directory should be the directory created by cloning this repository. Unzip the tarball from the Zenodo repository so that the working directory contains both `~/notebooks` and `~/data` subdirectories.

#### Download GRDC Streamflow Observation data
Unfortunately, due to licensing restrictions, we are not allowed to share GRDC streamflow observation data. This means that you will have to go to the GRDC website (url below) and download the GRDC observation data yourself. You can download data for specific gauges (choose the ‘Download by Station’ option on the GRDC webpage). The result of this download is a singel netCDF file with data from all requested gauges. The list of gauges used in this study is contained in the file `~/data/metadata/grdc_filtered.txt`, and can be input directly into the download query on the GRDC website.

Once you have downloaded the data, make sure that all data for all guages is contained in a single netCDF file named `GRDC-Daily.nc`. Place that file in the `~/data/grdc_data/GRDC-Daily.nc` or point to the file in the path variable `GRDC_DATA_FILE` in `~/notebooks/backend/data_paths.py`.

GRDC Data Portal: https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser

### Step 1: Define Working Directory in Source Code
In the file `~/backend/data_paths.py` change the local variable `_WORKING_DIR` to your current working directory. 

### Step 2: Create Cross Validation Gauge Splits
This step is not necessary to recreate the figures and analysis contained in the paper, but it is included in this repository for reproducibility. 

Groups of streamflow gauges that were used for cross validation studies are contained in the directory `~/data/gauge_groups_for_paper`. Code that shows how these cross validation splits were constructed is contained in the `1-create_ungauged_experiments_gauge_groups.ipynb` notebook in this directory. This notebook will produce two products:

1) Gauge groups as text files for various types of cross validation splits.
2) Maps of the locations of gauges in each cross validation split.

You have the option to create gauge splits with GRDC gauges and Caravan gauges (either or both combined).

Note that if you run this notebook it will overwrite any existing gauge groups with new ones. These new gauge groups will not be the same as the ones used in the paper, and since at least some of these gauge groups were created with a random number generator (i.e., the k-fold cross validation splits), using gauge groups that you create yourself instead of the ones that are in the `~/data/gauge_groups_for_paper` subdirectory will result in inaccurate statistics. Doing so will cause the AI model to appear better than it really is since results will be pulled from gauges that were not withheld during training. This notebook is included in this repository only so that you can see how the gauge groups were created.

### Step 3: Extract GloFAS Data for Gauges
This step is not necessary to recreate the figures and analysis contained in the paper, but is included in this repository for reproducibility.

GloFAS reanalysis data has already been extracted into netCDF files separately for each stream gauge. These data are stored in `~/data/model_data/glofas`. The notebook `2-extract_glofas_raw_data_into_netcdfs.ipynb` was used to extract GloFAS data from its native GRIB file format into per-basin netCDF files. You do not need to run this notebook, it is just here to demonstrate how this extraction was performed for the study. Notice that this extraction is non-trivial because of the method for matching GloFAS pixels to GRDC gauges.

### Step 4: Calculate Hydrograph Skill Scores
Once you have the GRDC netCDF file in the correct location within your directory structure, run the `3-calculate_standard_hydrograph_metrics.ipynb` notebook to calculate a set of standard hydrological skill metrics on modeled hydrographs. This notebook produces plots that are in the paper’s Supplementary Material.

### Step 5: Calculate Return Period Precision and Recall Metrics
Run the `4a-calculate_return_period_metrics.ipynb` notebook to calculate precision and recall metrics on different magnitude extreme events from modeled hydrographs.

Run the `4b-collect_return_period_metrics.ipynb` notebook to condense the return period metrics into a small number of pickle files to make them easier to load in the analysis notebooks. This step should probably be part of the `calculate_return_period_metrics.ipynb` notebook, but it is not -- I apologize for that. 

### Step 6: Create Figures and Results from Paper
Run the various `figure*.ipynb` notebooks to create figures from the paper. Figures that are shown in Supplementary Material are created within various notebooks throughout the steps described in this document.
