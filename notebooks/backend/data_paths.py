# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paths to all data for global model experiments."""

import pathlib


_WORKING_DIR = pathlib.Path('/home/gsnearing/data')

# --- Experiment Definition ----------------------------------------------------

LEAD_TIMES = list(range(8))

UNGAUGED_EXPERIMENTS = [
    'kfold_splits',
    'continent_splits',
    'climate_splits',
]
EXPERIMENTS = UNGAUGED_EXPERIMENTS + ['full_run']

GOOGLE_MODEL = 'dual_lstm'

# --- Saved Figures ------------------------------------------------------------

FIGURES_DIR = _WORKING_DIR / 'results_figures'

# --- Gauge Groups -------------------------------------------------------------

GAUGE_GROUPS_DIR = _WORKING_DIR / 'gauge_groups' / GOOGLE_MODEL
ADD_BACK_GAUGE_GROUP_DIR = GAUGE_GROUPS_DIR / 'data_addition_experiments'

# ---- Metadata ----------------------------------------------------------------
# This is for handling the metadata files supplied by ECMWF with the GloFAS
# data. This metadata contains three things that we need:
#   1) Match GRDC gauges with GloFAS pixels
#   2) Provides the GloFAS evaluation period
#   3) Provides the NSE and KGE scores that ECMWF calculated, which we use to
#      sanity check the scores we calculate on the same data.

METADATA_DIR = _WORKING_DIR / 'metadata'

# Full gauge group file.
FULL_GAUGE_GROUP_FILE = METADATA_DIR / 'grdc_filtered.txt'

# GloFAS Metadata.
GLOFAS_METADATA_FILE = METADATA_DIR / 'glofas_metadata.csv'
GLOFAS_STATION_INFO_FILE = METADATA_DIR / 'GloFAS_v3.2_oper_station_info-GloFAS_v3.2_oper_station_info.csv'
GLOFAS_SUPPLEMENTARY_INFO_FILE = METADATA_DIR / 'Supplementary_Information.xlsx-Qgis3.csv'
GLOFAS_v4_METADATA_FILE = METADATA_DIR / 'GRDC_GloFASv4_additional_information.csv'
GLOFAS_UPSREAM_AREA_FILE = METADATA_DIR / 'glofas_upArea.nc'

# Basin Attributes.
BASIN_ATTRIBUTES_FILE = METADATA_DIR / 'basin_attributes.csv'
GAUGE_COUNTRY_FILE = METADATA_DIR / 'basin_county.csv'
HYDROATLAS_COUNTRIES_PATH = METADATA_DIR / 'hybas_country_list.csv'
FULL_HYDROATLAS_ATTRIBUTES_FILENAME = METADATA_DIR / 'hydro_atlas_attributes.csv'

# GDP Data.
GDP_DATA_FILENAME = METADATA_DIR / 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4676807.csv'
PER_CAPITA_GDP_DATA_FILENAME = METADATA_DIR / 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5173062.csv'

# Country polygons for plotting.
COUNTRY_POLYGONS_FILENAME = METADATA_DIR / 'naturalearth_lowres.geojson'

# Gauge Dataset Metadata.
HYBAS_INFO_FILE = METADATA_DIR / 'hybas_gauges_info_lev12.csv'
HYBAS_ATTRIBUTES_FILE = METADATA_DIR / 'hydro_atlas_attributes_lev12.csv'
GRDC_METADATA_FILE = METADATA_DIR / 'grdc_stations_20220320.csv'

# Metadata synthesized by our workbooks.
GRDC_RECORD_LENGTH_FILENAME = METADATA_DIR / 'grdc_caravan_record_lengths.csv'


# --- Raw Model Runs -----------------------------------------------------------
# This is where the raw model data is stored in its native format.

GLOFAS_RAW_REANALYSIS_DATA_PATH = _WORKING_DIR / 'glofas_raw_data' / 'glofas_reanalysis'
GLOFAS_v4_RAW_REANALYSIS_DATA_PATH = _WORKING_DIR / 'glofas_raw_data' / 'GloFASv4.0_discharge_01011981_31122019.csv'

GOOGLE_MODEL_RUN_DIR = _WORKING_DIR / 'model_runs' / GOOGLE_MODEL

GRDC_DATA_FILE = _WORKING_DIR / 'grdc_data' / 'GRDC-Daily.nc'

# --- Extracted Model Runs -----------------------------------------------------
# Our workflow starts by standardizing the format of the model output that are
# stored in directories above, into per-gauge NetCDF files. This is faster
# and easier to load for calculating per-gauge metrics.

GOOGLE_EXTRACTED_RUNS_DIR = _WORKING_DIR / 'model_data' / 'google' / GOOGLE_MODEL
GOOGLE_EXTRACTED_RUNS_NO_GRDC_DIR = GOOGLE_EXTRACTED_RUNS_DIR / 'no_grdc_data'

GLOFAS_EXTRACTED_REFORECASTS_DIR = _WORKING_DIR / 'model_data' / 'glofas'  / 'reforecasts'
GLOFAS_EXTRACTED_REANALYSIS_DIR = _WORKING_DIR / 'model_data' / 'glofas' / 'reanalysis'
GLOFAS_v4_EXTRACTED_REANALYSIS_DIR = _WORKING_DIR / 'model_data' / 'glofas_v4' / 'reanalysis'
GLOFAS_v4_FULL_GRDC_EXTRACTED_REANALYSIS_DIR = _WORKING_DIR / 'model_data' / 'glofas_all_grdc' / 'reanalysis'

# The breakpoint directory is used for temporary storage during the process
# of extracting the raw GloFAS data into per-gauge NetCDF files. These
# breakpoint files store a subset of the time series for each gauge, which are
# concatenated later. This lets us restart the extraction workflow if it stops.
# This extraction workflow takes 10+ hours to complete.
GLOFAS_BREAKPOINTS_DIR = _WORKING_DIR / 'model_data' / 'glofas' / 'breakpoints'

# --- Experimental Results -----------------------------------------------------
# This is where we store the calculated statistics.

METRICS_DIR = _WORKING_DIR / 'metrics'

# Hydrograph metrics.
PER_GAUGE_GOOGLE_2014_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'google' / '2014' / GOOGLE_MODEL
PER_GAUGE_GOOGLE_1980_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'google' / '1980' / GOOGLE_MODEL
PER_GAUGE_GOOGLE_v3_PERIOD_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'google' / 'v3_benchmarking' / GOOGLE_MODEL
PER_GAUGE_GOOGLE_v4_PERIOD_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'google' / 'v4_benchmarking' / GOOGLE_MODEL

PER_METRIC_GOOGLE_2014_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'google' / '2014' / GOOGLE_MODEL
PER_METRIC_GOOGLE_1980_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'google' / '1980' / GOOGLE_MODEL
PER_METRIC_GOOGLE_v3_PERIOD_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'google' / 'v3_benchmarking' / GOOGLE_MODEL
PER_METRIC_GOOGLE_v4_PERIOD_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'google' / 'v4_benchmarking' / GOOGLE_MODEL

PER_GAUGE_GLOFAS_v3_REANALYSIS_BENCHMARKING_GAUGES_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'glofas_v3' / 'v3_benchmarking'
PER_GAUGE_GLOFAS_v3_REANALYSIS_ALL_GRDC_GAUGES_2014_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'glofas_v3' / '2014'
PER_GAUGE_GLOFAS_v3_REANALYSIS_ALL_GRDC_GAUGES_1980_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'glofas_v3' / '1980'

PER_METRIC_GLOFAS_v3_REANALYSIS_BENCHMARKING_GAUGES_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'glofas_v3' / 'v3_benchmarking'
PER_METRIC_GLOFAS_v3_REANALYSIS_ALL_GRDC_GAUGES_2014_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'glofas_v3' / '2014'
PER_METRIC_GLOFAS_v3_REANALYSIS_ALL_GRDC_GAUGES_1980_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'glofas_v3' / '1980'

PER_GAUGE_GLOFAS_v4_REANALYSIS_BENCHMARKING_GAUGES_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'glofas_v4' / 'v4_benchmarking'
PER_METRIC_GLOFAS_v4_REANALYSIS_BENCHMARKING_GAUGES_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'glofas_v4' / 'v4_benchmarking'


# Return period metrics.
GOOGLE_2014_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'google' / '2014' / GOOGLE_MODEL
GOOGLE_1980_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'google' / '1980' / GOOGLE_MODEL
GOOGLE_v3_PERIOD_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'google' / 'v3_benchmarking' / GOOGLE_MODEL
GOOGLE_v4_PERIOD_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'google' / 'v4_benchmarking' / GOOGLE_MODEL

GLOFAS_v3_REANALYSIS_2014_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'glofas_v3' / '2014'
GLOFAS_v3_REANALYSIS_1980_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'glofas_v3' / '1980'
GLOFAS_v3_REANALYSIS_v3_PERIOD_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'glofas_v3' / 'v3_benchmarking'
GLOFAS_v3_REANALYSIS_v4_PERIOD_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'glofas_v3' / 'v4_benchmarking'

GLOFAS_v4_REANALYSIS_v4_PERIOD_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'glofas_v4' / 'v4_benchmarking'

CONCATENATED_RETURN_PERIOD_DICTS_DIR = METRICS_DIR / 'concatenated_return_period_metrics'

# --- Figure File Names --------------------------------------------------------------------

GDP_GRDC_RECORD_LENGTH_CORRELATION_FILENAME = 'gdp_grdc_record_length_correlation.png'
GLOBAL_F1_SCORES_MAP_FILENAME = 'global_f1_scores_map.png'
RETURN_PERIOD_RELIABILITY_DISTRIBUTIONS_FILENAME = 'return_period_reliability_distributions.png'
LEAD_TIME_RELIABILITY_DISTRIBUTIONS_FILENAME = 'lead_time_reliability_distributions.png'
CONTINENT_RELIABILITY_SCORES_DISTRIBUTIONS_FILENAME = 'continent_reliability_scores_distributions.png'
PREDICTABILITY_WHICH_MODEL_IS_BETTER = 'whch_model_where.png'
PREDICATABILITY_CONFUSION_MATRICES_FILENAME = 'score_prediction_confusion_matrices.png'
GLOBAL_PREDICTED_SKILL_MAP_FILENAME = 'global_predicted_skill_map.png'

# Supplementary material.
ALL_GAUGE_LOCATIONS_MAP_FIGURE_FILENAME = 'gauge_locations_map.png'
EVALUATION_GAUGE_LOCATIONS_MAP_FIGURE_FILENAME = 'evaluation_gauge_locations_map.png'
GLOFAS_CALIBRATION_GAUGE_LOCATIONS_MAP_FIGURE_FILENAME = 'glofas_calibration_gauge_locations_map.png'
CROSS_VALIDATION_SPLITS_MAP_FIGURE_FILENAME = 'cross_validation_splits_map.png'
PREDICTABILITY_FEATURE_IMPORTANCES_FILENAME = 'prediction_feature_importances.png'

HYDROGRAPH_METRICS_WITH_LEAD_TIMES_CDFS_FILENAME = 'hydrograph_metrics_with_lead_times_cdfs.png'
HYDROGRAPH_METRICS_GLOFAS_UNGAUGED_CDFS_FILENAME = 'hydrograph_metrics_glofas_v3_ungauged_cdfs.png'
HYDROGRAPH_METRICS_GLOFAS_GAUGED_CDFS_FILENAME = 'hydrograph_metrics_glofas_v3_gauged_cdfs.png'
HYDROGRAPH_METRICS_v4_GLOFAS_GAUGED_CDFS_FILENAME = 'hydrograph_metrics_glofas_v4_gauged_cdfs.png'
