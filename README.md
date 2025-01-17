# AI Increases Global Access to Reliable Flood Forecasts

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository allows you to recreate the figures and statistics from the following paper:

[Nearing, Grey, et al. "Global prediction of extreme floods in ungauged watersheds." Nature (2024).](<https://www.nature.com/articles/s41586-024-07145-1>)


## Table of Contents
- [Overview](#overview)
- [License](#license)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Documentation](#documentation)
- [Issues](https://github.com/googlestaging/global_streamflow_model_paper/issues)

-----

## Overview
The code in this repository is structured so that all analysis can be done with python notebooks in the `~/notebooks` directory. The expected runtime is approxiamtely one day for the full analysis. The steps are as follows:

1) Download model data, metadata, and pre-calculated metrics from the associated Zenodo repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10397664.svg)](https://doi.org/10.5281/zenodo.10397664).

2) Download and prepare Global Runoff Data Center (GRDC) streamflow observation data and model simulation data. This step is not necessary if you want to use the pre-calculated statistics included in the Zenodo repository.

3) Run notebooks to calclate metrics. This step is not necessary if you want to use the pre-calculated statistics included in the Zenodo repository.

4) Run notebooks to produce figures and analyses.

Detailed instructions for these three steps are below.

Also included in the `~/notebooks` directory is a subdirectory called 'backend', which contains much of the active (mostly functional) code used by the analysis notebooks. The user should only touch the source code in this directory to change their local working directory, as described in the instructions below. 

Within the `~/notebooks/backend` source directory is another source directory called `return_period_calculator`. This subdirectory contains python code for fitting return period distributions and estimating return periods. These calculations are based loosely on guidelines outlined in the USGS Bulletin 17c, with some differences related to the statistical tests used for identifying outliers.

## License
This repository is licensed under an Apache 2.0 open source license. Please see the [LICENSE](https://github.com/googlestaging/global_streamflow_model_paper/blob/main/LICENSE) file in the root directory for the full license. 

This is not an official Google product.

## System Requirements
This repository should run on any computer and operating system that supports Python version 3. It has been tested on the Debian GNU/Linux 11 (bullseye) operating system. Running the notebooks for calculating metrics requires 128 GB of local memory.

## Installation
No software installation is required beyond Python v3 and Python libraries contained in the environment file. This repository is based on Python notebooks and can be run directly from a local clone:

`git clone https://github.com/googlestaging/global_streamflow_model_paper.git`

An [environment file](https://github.com/googlestaging/global_streamflow_model_paper/blob/main/environment.yml) is included for installing the necessary Python dependencies. If you are using Anaconda (miniconda, etc.) you may create this invironment with the following command from inside the `global_streamflow_model_paper` directory that results from cloning this repository:

`conda env create -f environment.yml`

## Documentation
Detailed Steps to Recreate Results Reported in the Paper

### Step 0: Define Working Directory in Source Code
In the file `~/notebooks/backend/data_paths.py` change the local variable `_WORKING_DIR` to your current working directory. 

### Step 1: Download Model Data

You will need to download and unzip/untar the tarballs from the Zenodo repository listed in the Code and Data Availability section of the paper referenced at the top of this README document. The DOI for the zenodo repository is: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10397664.svg)](https://doi.org/10.5281/zenodo.10397664)

Your working directory should be the directory created by cloning this repository. Unpacking the tarballs from the Zenodo repository will result in the following subdirectories: `~/model_data`, `~/metadata`, and `~/metrics`, and `~/gauge_groups_for_paper`. All of these subdirectories should be placed in the working directory so that the working directory contains `~/notebooks` (and other subdirectories included in this Github repository), as well as `~/model_data` (and all other subdirectories from the Zenodo repository). 

The model output data in this repository include reforecasts from the Google model and reanalyses from the GloFAS model. Google model outputs are in units [mm/day] and GloFAS outputs are in units [m3/s]. Modlel outputs are daily and timestamps are right-labeled, meaning that model ouputs labeled, .e.g., 01/01/2020 correspond to streamflow predictions for the day of 12/31/2019. 

### (Not Required) Step 2: Download GRDC Streamflow Observation Data
Due to licensing restrictions, we are not allowed to share streamflow observation data from the Global Runoff Data Center (GRDC). Using the [GRDC Data Portal](https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser), download GRDC data for all stations that are listed in the `~/gauge_groups/dual_lstm/grdc_filtered.txt` file. Download these as daily NetCDF files. This requires registering with the GRDC. You will likely have to download these data in multiple batches, resulting in multiple NetCDF files. If that is the case, name each of the NetCDF files uniqely and put them into a single directory somewhere on your local machine. Point to that directory using the `GRDC_DATA_DOWNLOAD_DIRECTORY` variable in the `~/notebooks/backend/data_paths.py` file, and then run the `~/notebooks/concatenate_grdc_downloads.ipynb` notebook to concatenate the download files into one netcdf file.

GRDC Data Portal: https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser

### (Not Required) Understand Cross Validation Gauge Splits
Groups of streamflow gauges that were used for cross validation studies are contained in the directory `~/gauge_groups_for_paper` (from the Zenodo repository). Code that shows how these cross validation splits were constructed is contained in the `~/notebooks/create_ungauged_experiments_gauge_groups.ipynb` notebook. This notebook will produce two products:

1) Gauge groups as text files for various types of cross validation splits, which are stored in `~/gauge_groups` directory.
2) Maps of the locations of gauges in each cross validation split.

You have the option to create gauge splits with GRDC gauges and Caravan gauges (either or both combined).

Note that if you run this notebook it will overwrite any existing gauge groups with new ones. These new gauge groups will not be the same as the ones used in the paper, since at least some of these gauge groups were created with a random number generator (i.e., the k-fold cross validation splits and the hydrologically-separated gauge splits). Using gauge groups that you create yourself instead of the ones that are in the `~/gauge_groups_for_paper` subdirectory will result in inaccurate statistics. Doing so will cause the AI model to appear better than it really is since results will be pulled from gauges that were not withheld during training. This notebook is included in this repository only so that you can see how the gauge groups were created.

### (Not Required) Step 3: Calculate Metrics
Once you have the GRDC netCDF file created, run the `/notebooks/calculate_standard_hydrograph_metrics.ipynb` notebook to calculate a set of standard hydrological skill metrics on modeled hydrographs. This notebook produces plots that are in the paper’s Supplementary Material.

Next, run the `~/notebooks/calculate_return_period_metrics.ipynb` notebook to calculate precision and recall metrics on different magnitude extreme events from modeled hydrographs. 

### Step 4: Create Figures and Results from Paper
Run the various `figure_*.ipynb` notebooks to create figures from the paper. These figures are saved in both PNG and vector graphics formats in the directory `~/results_figures`
