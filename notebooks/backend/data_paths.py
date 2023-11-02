"""Paths to all data for global model experiments."""

import pathlib

_WORKING_DIR = pathlib.Path('/home/gsnearing/github_repo/global_streamflow_model_paper')

# --- Experiment Definition ----------------------------------------------------

LEAD_TIMES = list(range(8))

UNGAUGED_EXPERIMENTS = [
    'kfold_splits',
    'continent_splits',
    'climate_splits',
    'hydrologically_separated',
]
EXPERIMENTS = UNGAUGED_EXPERIMENTS + ['full_run']

GOOGLE_MODEL = 'dual_lstm'

# --- Gauge Groups -------------------------------------------------------------

GAUGE_GROUPS_DIR = _WORKING_DIR / 'gauge_groups' / GOOGLE_MODEL
FULL_GAUGE_GROUP_FILE = GAUGE_GROUPS_DIR / 'grdc_filtered.txt'

# --- Data Paths ---------------------------------------------------------------

GOOGLE_MODEL_RUNS_DIR = _WORKING_DIR / 'model_data' / 'google' / GOOGLE_MODEL
GLOFAS_MODEL_RUNS = _WORKING_DIR / 'model_data' / 'GRDCstattions_GloFASv40' / 'dis24h_GLOFAS4.0_3arcmin_197901-202212_statsgoogle20230918.nc'

GRDC_DATA_DOWNLOAD_DIRECTORY = pathlib.Path('/home/gsnearing/data/grdc_data')
GRDC_DATA_FILE = _WORKING_DIR / 'grdc_data' / 'GRDC-Daily.nc'

# ---- Metadata ----------------------------------------------------------------

METADATA_DIR = _WORKING_DIR / 'metadata'

# Basin Attributes.
BASIN_ATTRIBUTES_FILE = METADATA_DIR / 'basin_attributes.csv'
GAUGE_COUNTRY_FILE = METADATA_DIR / 'basin_county.csv'

# GDP Data.
GDP_DATA_FILENAME = METADATA_DIR / 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4676807.csv'

# Country polygons for plotting.
COUNTRY_POLYGONS_FILENAME = METADATA_DIR / 'naturalearth_lowres.geojson'

# Metadata synthesized by our workbooks.
GRDC_RECORD_LENGTH_FILENAME = METADATA_DIR / 'grdc_caravan_record_lengths.csv'

# GLOFAS Metadata
GLOFAS_METADATA_FILENAME = _WORKING_DIR / 'model_data' / 'GRDCstattions_GloFASv40' / 'outlets_GloFASv4.0_google20230918.csv'
GLOFAS_v4_GAUGED_METADATA_FILENAME = METADATA_DIR / 'GRDC_GloFASv4_additional_information.csv'

# HydroATLAS Metadata
FULL_HYDROATLAS_ATTRIBUTES_FILENAME = METADATA_DIR / 'hydro_atlas_attributes.csv'
HYBAS_INFO_FILE = METADATA_DIR / 'hybas_gauges_info_lev12.csv'
HYBAS_ATTRIBUTES_FILE = METADATA_DIR / 'hydro_atlas_attributes_lev12.csv'
GRDC_METADATA_FILE = METADATA_DIR / 'grdc_stations_20220320.csv'
HYDROATLAS_COUNTRIES_PATH = METADATA_DIR / 'hybas_country_list.csv'

# --- Experimental Results -----------------------------------------------------

METRICS_DIR = _WORKING_DIR / 'metrics'

# Hydrograph metrics.
PER_GAUGE_GOOGLE_2014_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'google' / '2014' / GOOGLE_MODEL
PER_GAUGE_GOOGLE_1980_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'google' / '1980' / GOOGLE_MODEL

PER_METRIC_GOOGLE_2014_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'google' / '2014' / GOOGLE_MODEL
PER_METRIC_GOOGLE_1980_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'google' / '1980' / GOOGLE_MODEL

PER_GAUGE_GLOFAS_2014_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'glofas' / '2014'
PER_GAUGE_GLOFAS_1980_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_gauge' / 'glofas' / '2014'

PER_METRIC_GLOFAS_2014_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'glofas' / '2014'
PER_METRIC_GLOFAS_1980_HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'glofas' / '1980'

# Return period metrics.
GOOGLE_2014_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'google' / '2014' / GOOGLE_MODEL
GOOGLE_1980_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'google' / '1980' / GOOGLE_MODEL
GLOFAS_2014_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'glofas' / '2014' / GOOGLE_MODEL
GLOFAS_1980_RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics' / 'glofas' / '1980' / GOOGLE_MODEL

CONCATENATED_RETURN_PERIOD_DICTS_DIR = METRICS_DIR / 'concatenated_return_period_metrics'

# --- Figure File Names --------------------------------------------------------------------

FIGURES_DIR = _WORKING_DIR / 'results_figures'

GDP_GRDC_RECORD_LENGTH_CORRELATION_FILENAME = 'gdp_grdc_record_length_correlation'
GLOBAL_F1_SCORES_MAP_FILENAME = 'global_f1_scores_map'
RETURN_PERIOD_RELIABILITY_DISTRIBUTIONS_FILENAME = 'return_period_reliability_distributions'
LEAD_TIME_RELIABILITY_DISTRIBUTIONS_FILENAME = 'lead_time_reliability_distributions'
CONTINENT_RELIABILITY_SCORES_DISTRIBUTIONS_FILENAME = 'continent_reliability_scores_distributions'
PREDICTABILITY_WHICH_MODEL_IS_BETTER = 'which_model_where'
PREDICATABILITY_CONFUSION_MATRICES_FILENAME = 'score_prediction_confusion_matrices'
GLOBAL_PREDICTED_SKILL_MAP_FILENAME = 'global_predicted_skill_map'

# Supplementary material.
ALL_GAUGE_LOCATIONS_MAP_FIGURE_FILENAME = 'gauge_locations_map'
CALVAL_GAUGE_LOCATIONS_MAP_FIGURE_FILENAME = 'calval_gauge_locations_map'
CROSS_VALIDATION_SPLITS_MAP_FIGURE_FILENAME = 'cross_validation_splits_map'
PREDICTABILITY_FEATURE_IMPORTANCES_FILENAME = 'prediction_feature_importances'

HYDROGRAPH_METRICS_WITH_LEAD_TIMES_CDFS_FILENAME = 'hydrograph_metrics'
HYDROGRAPH_METRICS_GLOFAS_CALIBRATED_CDFS_FILENAME = 'hydrograph_metrics_glofas_calibrated'