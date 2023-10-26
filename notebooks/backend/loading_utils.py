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

"""Utilities for loading files for global model experiments."""

import os
import pathlib
import pickle as pkl
from typing import Optional

# h5netcdf is necessary as a backend for xarray load_file().
# It is necessary to have this, and importing it explicitly
# ensures that it is not removed from BUILD and stays visible.
# This is likely not necessary if running outside of Google.
import h5netcdf  # pylint: disable=unused-import
import numpy as np
import pandas as pd
import tqdm
import xarray

from backend import data_paths
from backend import metrics_utils

GLOFAS_REANALYSIS_VARIABLE_NAME  = 'glofas_reanalysis'
GLOFAS_REFORECASTS_VARIABLE_NAME  = 'glofas_reforecasts'

_GLOFAS_SEARCH_MULTIPLIER = 1
_ALLOWED_UPSTREAM_AREA_DIFFERENCE_FRACTION = 0.1
_MIN_SEARCH_RADIUS_KM = 21 


def create_remote_folder_if_necessary(
    path_to_create: pathlib.Path
) -> None:
  """Creates a remote directory."""
  if os.path.exists(path_to_create):
    return
  os.makedirs(path_to_create)

# -- Utilities for loading metadata files --------------------------------------


def prepare_and_save_glofas_v3_metadata() -> pd.DataFrame:

  # Load GloFAS Station Info file. This has the GRDC gauge ID.
  with open(data_paths.GLOFAS_STATION_INFO_FILE, 'r') as f:
    glofas_station_info_df = pd.read_csv(f)

  def get_grdc_id(provider_1, provider_2, provider_id_1, provider_id_2):
    if provider_1 == 'GRDC':
      return provider_id_1
    if provider_2 == 'GRDC':
      return provider_id_2
    return 'NOT_GRDC'

  # Pull basin IDs for all GRDC basins. The GRDC can be either the first or
  # second provider, and we need the ID from GRDC and not a different provider.
  glofas_station_info_df['grdc_id'] = glofas_station_info_df.apply(
      lambda x: get_grdc_id(x.Provider_1, x.Provider_2, x.ProvID_1, x.ProvID_2),
      axis=1
  )

  # Pull only GRDC basins from the dataframe.
  glofas_station_info_df = glofas_station_info_df.query('grdc_id != "NOT_GRDC"')

  # Pull only the columns we need.
  columns_to_keep = ['station_id', 'grdc_id']
  glofas_station_info_df = glofas_station_info_df[columns_to_keep]

  # The Supplementary Info file has information about calibration and
  # validation, which allows us to compare metrics that we calculate against
  # metrics that ECMWF calculated.
  #
  # Note that both files have a latitude and longitue for the GloFAS pixel
  # associated with each gauge ID -- in the Station Info file, these columns
  # are LisfloodX and LisfloodY, while here the columns are lat_GloFAS and
  # long_GloFAS. These pixel locations do not always agree between the two
  # files, so we chose to use the one from this file, since we will compare
  # against the metrics that are supplied in this file.
  with open(data_paths.GLOFAS_SUPPLEMENTARY_INFO_FILE, 'r') as f:
    glofas_supplementary_info_df = pd.read_csv(f)
  glofas_supplementary_info_df = glofas_supplementary_info_df.query(
      'Provider == "GRDC"'
  )
  columns_to_keep = [
      'ID',
      'latitude',
      'longitude',
      'lat_GloFAS',
      'long_GloFAS',
      'Validation_Start',
      'Validation_End',
      'Calibration_Start',
      'Calibration_End',
      'KGE_cal',
      'KGE_val',
      'NSE_cal',
      'NSE_val',
  ]
  glofas_supplementary_info_df = glofas_supplementary_info_df[columns_to_keep]

  # Merge glofas_station_info_df and glofas_supplementary_info_df on the
  # Glofas_ID field. Then we will get the grdc id from glofas_station_info_df,
  # with the Glofas lat/long location from glofas_supplementary_info_df.
  # This is what we need to understand their matching of GRDC_ID <--> model
  # predictions in the correct pixel.
  ids_1 = glofas_station_info_df['station_id']
  ids_2 = glofas_supplementary_info_df['ID']
  assert len(ids_1) == len(set(ids_1))
  assert len(ids_2) == len(set(ids_2))
  assert len(set(ids_2) - set(ids_1)) == 0  # df_1 includes all gauges in df_2

  glofas_metadata = pd.merge(
      glofas_supplementary_info_df.set_index('ID', drop=True),
      glofas_station_info_df.set_index('station_id', drop=True),
      how='left',
      left_index=True,
      right_index=True
  )
  glofas_metadata.set_index('grdc_id', inplace=True)

  glofas_metadata.index = [f'GRDC_{idx}' for idx in glofas_metadata.index]

  with open(data_paths.GLOFAS_METADATA_FILE, 'w') as f:
    glofas_metadata.to_csv(f)

  return glofas_metadata


def load_full_hydroatlas_attributes_file() -> pd.DataFrame:

  with open(data_paths.FULL_HYDROATLAS_ATTRIBUTES_FILENAME, 'rt') as f:
    hybas_attributes = pd.read_csv(f, index_col='Unnamed: 0')
    
  column_mapper = {col: col.split('HYDRO_ATLAS:')[1] for col in hybas_attributes.columns}
  hybas_attributes = hybas_attributes.rename(columns=column_mapper)

  return hybas_attributes


def load_grdc_record_length_file() -> pd.DataFrame:
  with open(data_paths.GRDC_RECORD_LENGTH_FILENAME, 'rt') as f:
    record_lengths_df = pd.read_csv(f, index_col='Gauge ID')
  return record_lengths_df


def load_hydroatlas_country_file() -> pd.DataFrame:
  with open(data_paths.HYDROATLAS_COUNTRIES_PATH, 'rt') as f:
    hydroatlas_countries = pd.read_csv(f)
  hydroatlas_countries['HyBAS ID'] = hydroatlas_countries[
      'HyBAS ID'].apply(lambda x: int(x.split('_')[1]))
  hydroatlas_countries.set_index('HyBAS ID', inplace=True)
  return hydroatlas_countries


def load_gdp_file() -> pd.DataFrame:
  with open(data_paths.GDP_DATA_FILENAME, 'rt') as f:
    gdp_df = pd.read_csv(f, header=2, index_col='Country Name')

  def get_most_recent_gdp_value(gdp_series):
    gdp_data = gdp_series[[idx for idx in gdp_series.index if idx.isdigit()]]
    gdp_data = gdp_data.values.astype(float)
    if all(np.isnan(gdp_data)):
      return np.nan
    else:
      return gdp_data[~np.isnan(gdp_data)][-1]

  # Create a Pandas series of GDP indexed by country name.
  most_recent_gdp = {
      country: get_most_recent_gdp_value(gdp_series)
      for country, gdp_series in gdp_df.iterrows()
  }
  most_recent_gdp = pd.Series(most_recent_gdp, name='Most Recent GDP')
  most_recent_gdp.index.rename('Country', inplace=True)

  # Rename all countries in GDP dataset to match names in HydroBasins dataset.
  most_recent_gdp.rename(index={'Iran, Islamic Rep.': 'Iran'}, inplace=True)
  most_recent_gdp.rename(index={'Russian Federation': 'Russia'}, inplace=True)
  most_recent_gdp.rename(index={'Slovak Republic': 'Slovakia'}, inplace=True)
  most_recent_gdp.rename(index={
      'Congo, Dem. Rep.': 'Democratic Republic of the Congo'}, inplace=True)
  most_recent_gdp.rename(index={"Cote d'Ivoire": "Côte d'Ivoire"}, inplace=True)
  most_recent_gdp.rename(index={
      'Sao Tome and Principe': 'São Tomé and Príncipe'}, inplace=True)
  most_recent_gdp.rename(index={'Kyrgyz Republic': 'Kyrgyzstan'}, inplace=True)
  most_recent_gdp.rename(index={'Korea, Rep.': 'South Korea'}, inplace=True)
  most_recent_gdp.rename(index={'Venezuela, RB': 'Venezuela'}, inplace=True)
  most_recent_gdp.rename(index={'Turkiye': 'Turkey'}, inplace=True)
  most_recent_gdp.rename(index={'Syrian Arab Republic': 'Syria'}, inplace=True)
  most_recent_gdp.rename(index={'Lao PDR': 'Laos'}, inplace=True)

  return most_recent_gdp


def load_grdc_record_length_file() -> pd.DataFrame:
  with open(data_paths.GRDC_RECORD_LENGTH_FILENAME, 'rt') as f:
    record_lengths_df = pd.read_csv(f, index_col='Gauge ID')
  return record_lengths_df


def load_gauge_country_file() -> pd.DataFrame:
  with open(data_paths.GAUGE_COUNTRY_FILE, 'rt') as f:
    gauge_countries = pd.read_csv(f, index_col='gauge_id')
  return gauge_countries


def load_hydroatlas_attributes_file() -> pd.DataFrame:
  with open(data_paths.HYBAS_ATTRIBUTES_FILE, 'rt') as f:
    hydroatlas_attributes = pd.read_csv(f)
  hydroatlas_attributes['HYBAS_ID'] = hydroatlas_attributes[
      'HYBAS_ID'].apply(lambda x: int(x.split('_')[1]))
  hydroatlas_attributes.set_index('HYBAS_ID', inplace=True)
  return hydroatlas_attributes


def load_hydroatlas_info_file() -> pd.DataFrame:
  with open(data_paths.HYBAS_INFO_FILE, 'rt') as f:
    return pd.read_csv(f, index_col='HYBAS_ID')


def load_glofas_metadata_file() -> pd.DataFrame:
  """Loads the GloFAS metadata file."""
  with open(data_paths.GLOFAS_METADATA_FILE, 'r') as f:
    return pd.read_csv(f, index_col='Unnamed: 0')


def load_glofas_v4_metadata_file() -> pd.DataFrame:
  """Loads the GloFAS metadata file."""
  with open(data_paths.GLOFAS_v4_METADATA_FILE, 'r') as f:
    metadata = pd.read_csv(f)
  metadata['grdc_number'] = metadata['grdc_number'].apply(lambda x: f'GRDC_{x}')
  metadata.set_index('grdc_number', inplace=True)
  return metadata

def load_attributes_file(
    gauges: Optional[list[str]] = None,
    attributes: Optional[list[str]] = None
) -> pd.DataFrame:
  """Loads a basin attributes file."""

  with open(data_paths.BASIN_ATTRIBUTES_FILE, 'r') as f:
    df = pd.read_csv(f, index_col='Unnamed: 0')

  if gauges is not None:
    df = df.loc[gauges]

  if attributes is not None:
    df = df[attributes]

  return df


def _load_netcdf(
    file_path: pathlib.Path
) -> xarray.Dataset:
  """Loads all data for a gauge from a NetCDF file.

  Args:
    file_path: Name of NetCDF file to load.

  Returns:
    Pandas DataFrame indexed by basin id with lead times in columns.
  """

  # Open the dataset.
  try:
    data = xarray.load_dataset(file_path)
  except Exception as e:
    return None

  return data


def _load_gauge(
    results_file: pathlib.Path,
    time_period: Optional[list[str]] = None,
    lead_times: Optional[list[int]] = None,
) -> xarray.Dataset:
  """Load a timeselice of model results from one gauge."""

  # Load the netcdf file.
  data = _load_netcdf(file_path=results_file)

  # If requested, pull only the specified times and lead times.
  if time_period is not None:
    if len(time_period) != 2:
      raise ValueError(f'Time period list must have exactly two values, '
                       f'received {len(time_period)}.')
    data = data.sel(time=slice(*time_period))

  if lead_times is not None:
    data = data.sel(lead_time=lead_times)

  return data


def save_gauge(
    gauge_xarray: xarray.Dataset,
    gauge: int,
    remote_dir: pathlib.Path,
    file_suffix: Optional[str] = None,
    file_prefix: Optional[str] = None,
) -> pathlib.Path:
  """Save forecast timeseries for a single gauge."""

  # Create a temporary local path for writing.
  filename = f'{gauge}'
  if file_suffix:
    filename = f'{filename}_{file_suffix}'
  if file_prefix:
    filename = f'{file_prefix}_{filename}'
  filename = f'{filename}.nc'
  remote_path = remote_dir / filename

  if not os.path.exists(remote_dir):
    os.makedirs(remote_dir)
    
  # Save file locally.
  gauge_xarray.to_netcdf(remote_path)

  return remote_path

# --- Utilities for Loading Raw Google Model Runs ------------------------------


def load_raw_google_model_run_for_one_gauge(
    experiment: str,
    gauge: str,
    ungauged_gauge_groups: dict[dict[str, str]],
):
  """Loads a whole Google model result from one gauge from the raw runs."""
  # Find the gauge in validation gauge groups.
  if experiment in data_paths.UNGAUGED_EXPERIMENTS:
    splits_with_gauge = [
        split
        for split, gauges in ungauged_gauge_groups[experiment].items()
        if gauge in gauges
    ]
    if len(splits_with_gauge) != 1:
      raise ValueError(
          f'Expected to find exactly 1 instance of gauge {gauge} '
          f'in {experiment}, instead found '
          f'{len(splits_with_gauge)}.'
      )
    split = splits_with_gauge[0]

  # This try/except is due to a code change that happened in the middle of
  # our runs and changed the output directory structure.
  try:

    # Ungauged runs have a split, but the full run doesn't.
    if experiment in data_paths.UNGAUGED_EXPERIMENTS:
      results_file = (
          data_paths.GOOGLE_MODEL_RUN_DIR
          / experiment
          / split
          / 'qgis_output'
          / f'{gauge}.p'
      )
    else:
      results_file = (
          data_paths.GOOGLE_MODEL_RUN_DIR
          / experiment
          / 'qgis_output'
          / f'{gauge}.p'
      )

    # Try loading the file.
    with open(results_file, 'rb') as f:
      data = pkl.load(f)

  except:

    if experiment in data_paths.UNGAUGED_EXPERIMENTS:
      results_file = (
          data_paths.GOOGLE_MODEL_RUN_DIR
          / experiment
          / split
          / 'evaluation'
          / 'qgis_output'
          / f'{gauge}.p'
      )
    else:
      results_file = (
          data_paths.GOOGLE_MODEL_RUN_DIR
          / experiment
          / 'evaluation'
          / 'qgis_output'
          / f'{gauge}.p'
      )
    # Try loading the file.
    with open(results_file, 'rb') as f:
      data = pkl.load(f)

  # Format the data as we expect to use it later. 
  # Drop unnecessary dimensions for the metrics calculator.
  data = data.rename({'prediction': 'google_prediction'})
  data = data.drop(labels='percentiles')
  data = data.drop_dims('percentile')

  return data


# --- Utilities for Loading Extracted Google Model Runs ------------------------


def load_google_model_for_one_gauge(
    experiment: str,
    gauge: str,
    load_without_grdc: bool = True
) -> xarray.Dataset:
  """Loads a whole Google model result from one gauge."""

  # Create results file from experiment and gauge. This is specific to the
  # Google model.
  if not load_without_grdc:
    results_file = data_paths.GOOGLE_EXTRACTED_RUNS_DIR / experiment / f'{gauge}.nc'
  else:
    results_file = data_paths.GOOGLE_EXTRACTED_RUNS_NO_GRDC_DIR / experiment / f'{gauge}.nc'
    
  # Loads the gauge's data.
  return _load_gauge(results_file=results_file)


def load_ungauged_experiment_model_runs(
    experiment: str,
    gauges: list[str],
    load_without_grdc: bool = True
) -> xarray.Dataset:
  """Loads all runs for one ungauged experiment."""

  print(f'Working on experiment: {experiment}')

  # Loads model runs for all gauges in an experiment.
  results_dict = {}
  for gauge in tqdm.tqdm(gauges):
    results_dict[gauge] = load_google_model_for_one_gauge(
        experiment=experiment,
        gauge=gauge,
        load_without_grdc=load_without_grdc,
    )

  # Concats into a single xarray dataset indexed by gauge.
  return xarray.concat(
      [ds for ds in results_dict.values() if ds is not None],
      dim='gauge_id'
  )


def load_all_experimental_model_runs(
    gauges: list[str],
    load_without_grdc: bool = True
) -> dict[str, xarray.Dataset]:
  """Loads all runs for all Google model experiments."""

  return {
      experiment: load_ungauged_experiment_model_runs(
          experiment=experiment,
          gauges=gauges,
          load_without_grdc=load_without_grdc
      ) for experiment in data_paths.EXPERIMENTS
  }

# --- Utilities for Loading Raw GloFAS Model Runs ------------------------------

_GLOFAS_METADATA_LATITUDE_COLUMN = 'lat_GloFAS'
_GLOFAS_METADATA_LONGITUDE_COLUMN = 'long_GloFAS'
_GLOFAS_METADATA_UPSTREAM_AREA_COLUMN = 'calculated_upstream_area'


def load_glofas_v4_csv() -> pd.DataFrame:
  with open(data_paths.GLOFAS_v4_RAW_REANALYSIS_DATA_PATH, 'rt') as f:
    v4data = pd.read_csv(f)

  v4meta = load_glofas_v4_metadata_file()
  column_mapper = {
      glofas: grdc for glofas, grdc in zip(v4meta['GLOFASid'], v4meta.index)}
  v4data.rename(columns=column_mapper, inplace=True)

  v4data.rename(columns={'End_of_timestep': 'time'}, inplace=True)
  v4data['time'] = v4data['time'].apply(lambda x: pd.to_datetime(
      x, format='%d/%m/%Y %H:%M'))
  v4data.set_index('time', inplace=True)

  return v4data


def load_glofas_reforecsat_grib(
    filepath: pathlib.Path,
    is_grib: bool = False,
    is_netcdf: bool = False
) -> xarray.Dataset:
  """Loads a timeselice of GloFAS results.

  This function first copies the data to a local path, since xarray cannot
  read on CNS.

  Args:
    filepath: Name of GRIB file to load.
    is_grib: Boolean to tell the routine to load grib (not netcdf)
    is_netcdf: Boolean to tell the routine to load netcdf (not grib)

  Returns:
    Pandas DataFrame indexed by basin id with lead times in columns.
  """

  # Can't be both GRIB and NetCDF.
  if is_grib and is_netcdf:
    raise ValueError('Cannot be both GRIB and NetCDF.')
  if not (is_grib or is_netcdf):
    raise ValueError('Must be either GRIB or NetCDF.')

  # Open the dataset
  if is_grib:
    try:
      data_cf = xarray.load_dataset(
          filepath,
          engine='cfgrib',
      )
    except:
      data_cf = xarray.load_dataset(
          filepath,
          engine='cfgrib',
          filter_by_keys={'dataType': 'cf'}
      )
  elif is_netcdf:
    data_cf = xarray.load_dataset(filepath)

  return data_cf


def extract_xarray_for_gauge(
    global_xr: xarray.Dataset,
    glofas_upstream_area_xr: xarray.Dataset,
    latitude: float,
    longitude: float,
    upstream_area: float
) -> Optional[xarray.Dataset]:
  """Extracts a portion of an Xarray dataset for given lat/lon."""

  # Find nearest gridcell to the specified lat/lon.
  nearest_gridcell = glofas_upstream_area_xr.sel(
      latitude=latitude, 
      longitude=longitude, 
      method='nearest'
  )
  nearest_lat = float(nearest_gridcell.latitude.values)
  nearest_lon = float(nearest_gridcell.longitude.values)

  # Define the search area in number of glofas grid cells.
  basin_search_radius_km = 1/2 * _GLOFAS_SEARCH_MULTIPLIER * np.sqrt(upstream_area)
  search_radius_km = np.min([_MIN_SEARCH_RADIUS_KM, basin_search_radius_km])
  search_radius_gridcells = int(np.round(search_radius_km/10))

  # Find all gridcells within the search area.
  LATITUDE_SPACING = 0.1
  LONGITUDE_SPACING = 0.1

  lat_lims = (
      nearest_lat - search_radius_gridcells*LATITUDE_SPACING, 
      nearest_lat + search_radius_gridcells*LATITUDE_SPACING, 
  )
  lats = np.linspace(lat_lims[0], lat_lims[1], 1+2*search_radius_gridcells)

  lon_lims = (
      nearest_lon - search_radius_gridcells*LONGITUDE_SPACING, 
      nearest_lon + search_radius_gridcells*LONGITUDE_SPACING, 
  )
  lons = np.linspace(lon_lims[0], lon_lims[1], 1+2*search_radius_gridcells)

  search_upstream_area = glofas_upstream_area_xr.sel(
      latitude=lats,
      longitude=lons,
      method='nearest'
  )

  # Find the glofas pixel within search area with closest drainage area.
  abs_difference = np.abs(search_upstream_area.uparea - upstream_area)
  min_diff = abs_difference.where(
      abs_difference==abs_difference.min(), drop=True).squeeze()

  # If multiple pixels are found, choose the one with the closest longitude.
  if min_diff.count().values > 1:
    if min_diff.latitude.shape and min_diff.longitude.shape:
      min_diff = min_diff.sel(
          longitude=longitude,
          latitude=latitude,
          method='nearest'
      )
    elif min_diff.latitude.shape:
      min_diff = min_diff.sel(
          latitude=latitude,
          method='nearest'
      )
    elif min_diff.longitude.shape:
      min_diff = min_diff.sel(
          longitude=longitude,
          method='nearest'
      )
  if min_diff.count().values < 1:
    return None

  # Only use this pixel if the area is within specified range.
  min_diff = min_diff.fillna(np.inf)
  frac_diff = float(min_diff.values) / upstream_area
  if frac_diff > _ALLOWED_UPSTREAM_AREA_DIFFERENCE_FRACTION:
    return None

  # Get the data array for just the selected pixel.
  min_diff_lat = float(min_diff.latitude.values)
  min_diff_lon = float(min_diff.longitude.values)
  glofas_extracted_pixel = global_xr.sel(
      latitude=min_diff_lat, 
      longitude=min_diff_lon, 
      method='nearest'
  )

  return glofas_extracted_pixel


def load_glofas_reforecasts_for_all_basins_at_timestep(
    file_path: pathlib.Path,
    gauge_locations: pd.DataFrame,
    glofas_upstream_area: xarray.Dataset,
    lat_col: str = _GLOFAS_METADATA_LATITUDE_COLUMN,
    lon_col: str = _GLOFAS_METADATA_LONGITUDE_COLUMN,
    area_col: str = _GLOFAS_METADATA_UPSTREAM_AREA_COLUMN,
) -> xarray.Dataset:
  """Extracts information for a basin list from a single GloFAS GRIB file.

  Args:
    file_path: Name of GRIB file to load.
    gauge_locations: Dataframe indexed by gauge ID that contains columns
      with the latitude and longitude of the corresponding GloFAS pixel.
    lat_col: Name of column in 'gauge_locations' dataframe that contains
      latitude information.
    lon_col: Name of column in 'gauge_locations' dataframe that contains
      longitude information.

  Returns:
    Xarray dataset with a dimension that includes gauge ID.
  """

  # Load the grib file.
  global_xr = load_glofas_reforecsat_grib(file_path, is_grib=True)

  # Extract pixels for gauges.
  gauge_dict = {
      gauge: extract_xarray_for_gauge(
          global_xr=global_xr,
          glofas_upstream_area_xr=glofas_upstream_area,
          latitude=gauge_locations.loc[gauge, lat_col],
          longitude=gauge_locations.loc[gauge, lon_col],
          upstream_area=gauge_locations.loc[gauge, area_col]
      )
      for gauge in gauge_locations.index
  }

  # Remove missing basins.
  filtered_xrs = {gauge: xr for gauge, xr in gauge_dict.items() if xr is not None}

  # Concatenate everything into a single xarray.
  return xarray.concat(
      filtered_xrs.values(),
      dim=pd.Index(filtered_xrs.keys(), name='gauge_id')
  )


def load_glofas_reforecsat_nectdf(
    filepath: str
) -> xarray.Dataset:
  """Loads a timeselice of GloFAS results.

  This function first copies the data to a local path, since xarray cannot
  read on CNS.

  Args:
    filepath: Name of netcdf file to load.

  Returns:
    Pandas DataFrame indexed by basin id with lead times in columns.
  """

  # Copy to local.
  local_path = _copy_to_local(remote_path=filepath)

  # Open the dataset
  data_cf = xarray.load_dataset(
      local_path,
      engine='cfgrib',
      filter_by_keys={'dataType': 'cf'}
  )

  # Delete the local file.
  gfile.DeleteRecursively(local_path)

  return data_cf


def load_timeseries_from_gribs(
    grib_files: list[str],
    glofas_metadata: pd.DataFrame,
    lat_col: str = _GLOFAS_METADATA_LATITUDE_COLUMN,
    lon_col: str = _GLOFAS_METADATA_LONGITUDE_COLUMN,
    area_col: str = _GLOFAS_METADATA_UPSTREAM_AREA_COLUMN,
) -> xarray.Dataset:
  """Loads a set of grib files to create a timeseries xarray."""

  # Load the file containing upstream drainage areas for all glofas pixels.
  glofas_upstream_area = xarray.load_dataset(data_paths.GLOFAS_UPSREAM_AREA_FILE) / 1e3 / 1e3

  timestep_xrs = []
  for grib_file in tqdm.tqdm(grib_files):
    timestep_xrs.append(
        load_glofas_reforecasts_for_all_basins_at_timestep(
            file_path=pathlib.Path(grib_file),
            gauge_locations=glofas_metadata,
            glofas_upstream_area=glofas_upstream_area,
            lat_col=lat_col,
            lon_col=lon_col,
            area_col=area_col,
        )
    )
    
  return xarray.concat(timestep_xrs, dim='time')


# --- Utilities for Loading Extracted GloFAS Model Runs ------------------------


def load_glofas_reforecasts_for_one_gauge(
    gauge: str,
    time_period: Optional[list[str]] = None,
    lead_times: Optional[list[int]] = None,
    all_grdc_gauges: bool = False,
) -> xarray.Dataset:
  """Loads a timeselice of GloFAS model results from one gauge."""

  # Create results file from experiment and gauge. This is specific to the
  # Google model.
  if not all_grdc_gauges:
    results_file = data_paths.GLOFAS_EXTRACTED_REFORECASTS_DIR / f'{gauge}.nc'
  else:
    results_file = data_paths.GLOFAS_FULL_GRDC_EXTRACTED_REFORECASTS_DIR / f'{gauge}.nc'

  # Loads the gauge's data.
  return _load_gauge(
      results_file=results_file,
      time_period=time_period,
      lead_times=lead_times,
  )

def load_glofas_reanalysis_for_one_gauge(
    gauge: str,
    time_period: Optional[list[str]] = None,
    lead_times: Optional[list[int]] = None,
    all_grdc_gauges: bool = False,
    v4: bool = False,
) -> xarray.Dataset:
  """Loads a timeselice of GloFAS model results from one gauge."""

  if all_grdc_gauges and v4:
    raise ValueError('We do not have v4 predictions for all GRDC gauges.')

  # Create results file from experiment and gauge. This is specific to the
  # Google model.
  if v4:
    results_file = data_paths.GLOFAS_v4_EXTRACTED_REANALYSIS_DIR / f'{gauge}.nc'
  else:
    results_file = data_paths.GLOFAS_EXTRACTED_REANALYSIS_DIR / f'{gauge}.nc'

  # Loads the gauge's data.
  return _load_gauge(
      results_file=results_file,
      time_period=time_period,
      lead_times=lead_times,
  )


def load_glofas_model_runs(
    gauges: list[str],
    reforecasts: bool = False,
    reanalysis: bool = False,
    v4: bool = False,
    time_periods: Optional[dict[str, list[str]]] = None,
    lead_times: Optional[list[int]] = None,
    all_grdc_gauges: bool = False,
) -> xarray.Dataset:
  """Loads all runs for one ungauged experiment."""

  if sum([reforecasts, reanalysis, v4]) != 1:
    raise ValueError('Must load exactly one of reforecasts, reanalyses, or v4.')

  if time_periods is None:
    time_periods = {gauge: None for gauge in gauges}

  # Loads model runs for all gauges in an experiment.
  if reforecasts:
    results_dict = {
        gauge: load_glofas_reforecasts_for_one_gauge(
            gauge=gauge,
            time_period=time_periods[gauge],
            lead_times=lead_times,
            all_grdc_gauges=all_grdc_gauges,
        ) for gauge in tqdm.tqdm(gauges)
    }
    variable_name = GLOFAS_REFORECASTS_VARIABLE_NAME
  elif reanalysis:
    results_dict = {
        gauge: load_glofas_reanalysis_for_one_gauge(
            gauge=gauge,
            time_period=time_periods[gauge],
            lead_times=lead_times,
            all_grdc_gauges=all_grdc_gauges,
        ) for gauge in tqdm.tqdm(gauges)
    }
    variable_name = GLOFAS_REANALYSIS_VARIABLE_NAME
  elif v4:
    results_dict = {
        gauge: load_glofas_reanalysis_for_one_gauge(
            gauge=gauge,
            time_period=time_periods[gauge],
            lead_times=lead_times,
            all_grdc_gauges=False,
            v4=True,
        ) for gauge in tqdm.tqdm(gauges)
    }
    variable_name = GLOFAS_REANALYSIS_VARIABLE_NAME
  else:
    raise ValueError('Must load exactly one of reforecasts, reanalyses, or v4.')

  # Concats into a single xarray dataset indexed by gauge.
  glofas_data = xarray.concat(
      [ds for ds in results_dict.values() if ds is not None],
      dim='gauge_id'
  )

  # Standardize some of the dim names.
  glofas_data['step'] = list(range(10))
  glofas_data = glofas_data.rename(
      {
          'step': 'lead_time',
          'dis24': variable_name
      }
  )

  return glofas_data

# --- Utilities for loading GRDC Observation data -------------------------------


def calculate_cms_to_mm_per_day_constant(drain_area: float) -> float:
  """Returns the constant to multiply by for this unit change.

  The 24 * 60 * 60 part translates from seconds to days.
  The 1000 in the nominator translates "1 of the 3" meter units in m**3/s to mm.
  The 1000**2 in the denominator is translating the area from km**2 to m**2.

  Args:
    drain_area: the drain area of the basin in square kilometers.
  """
  return (24 * 60 * 60 * 1000) / (drain_area * 1e6)


def unnormalize_observation(
    normalized_discharge: xarray.DataArray,
) -> xarray.DataArray:
    
  # Create a series of drainage areas.
  drain_area_column_name = 'calculated_drain_area'
  drain_area = load_attributes_file(attributes=[drain_area_column_name])
    
  # Calculate the multiplier for this gauge area - to convert from cms to mm.
  drain_area_multiplier = drain_area[
      drain_area_column_name].apply(
      lambda x: calculate_cms_to_mm_per_day_constant(x))
  drain_area_multiplier = drain_area_multiplier.squeeze().to_xarray().rename({'index': 'gauge_id'})

  # Create the unnormalized discharge variable
  unnormalized_discharge = normalized_discharge / drain_area_multiplier
  unnormalized_discharge = unnormalized_discharge.rename(metrics_utils.UNNORMALIZED_OBS_VARIABLE)

  return unnormalized_discharge


def normalize_observation(
    unnormalized_discharge: xarray.DataArray,
#     drain_area: Optional[dict] = None
) -> xarray.DataArray:
    
  # Create a series of drainage areas.
  drain_area_column_name = 'calculated_drain_area'
  drain_area = load_attributes_file(attributes=[drain_area_column_name])
  
  # Calculate the multiplier for this gauge area - to convert from cms to mm.
  drain_area_multiplier = drain_area[
      drain_area_column_name].apply(
      lambda x: calculate_cms_to_mm_per_day_constant(x))
  drain_area_multiplier = drain_area_multiplier.squeeze().to_xarray().rename({'index': 'gauge_id'})

#   drain_area_multiplier = {
#     gauge_id: calculate_cms_to_mm_per_day_constant(area)
#     for gauge_id, area in drain_area.items()
#   }
#   drain_area_multiplier = pd.Series(drain_area_multiplier)
#   drain_area_multiplier = xarray.DataArray(
#       drain_area_multiplier.values,
#       dims=['gauge_id'],
#       coords={'gauge_id': drain_area_multiplier.keys()}
#   )
    
  # Create the unnormalized discharge variable
  normalized_discharge = unnormalized_discharge * drain_area_multiplier
  normalized_discharge = normalized_discharge.rename(metrics_utils.OBS_VARIABLE)

  return normalized_discharge


def load_grdc_data() -> xarray.Dataset:
  """Loads a single file of GRDC data."""

  # Load the nectdf file.
  grdc_observation_data = xarray.load_dataset(
    data_paths.GRDC_DATA_FILE,
    engine='netcdf4'
  )

  # Name the runoff variable.
  grdc_observation_data = grdc_observation_data.rename({'runoff_mean': metrics_utils.UNNORMALIZED_OBS_VARIABLE})

  # Rename the gauge id coordinate and use the same gauge_ids as the rest of this codebase.
  grdc_observation_data = grdc_observation_data.rename({'id': 'gauge_id'})
  grdc_observation_data['gauge_id'] = [f'GRDC_{id}' for id in grdc_observation_data.gauge_id.values]

  # Replace missing values with NaNs.
  grdc_observation_data[
      metrics_utils.UNNORMALIZED_OBS_VARIABLE] = grdc_observation_data[
      metrics_utils.UNNORMALIZED_OBS_VARIABLE].where(
      grdc_observation_data[metrics_utils.UNNORMALIZED_OBS_VARIABLE] > 0)

  # Extract area for later.
  area = grdc_observation_data['area']
    
  # Add lead time.
  lead_time_data = grdc_observation_data[metrics_utils.UNNORMALIZED_OBS_VARIABLE].values
  lead_time_data = np.stack([lead_time_data]*10, -1)
  for lt in range(1, 9):
    lead_time_data[:-lt, :, lt] = lead_time_data[lt:, :, lt]
    
  # Shift by one day to account for the fact that we right-label all data from the models.
  for lt in range(9):
    lead_time_data[1:, :, lt] = lead_time_data[:-1, :, lt]

  # Put the data back into an xarray. This time without a lot of the attributes that we don't need.
  grdc_observation_data = xarray.Dataset(
    data_vars={
        metrics_utils.UNNORMALIZED_OBS_VARIABLE: (['time', 'gauge_id', 'lead_time'], lead_time_data)
    },
    coords={
        'time': (['time'], grdc_observation_data.time.values),
        'gauge_id': (['gauge_id'], grdc_observation_data.gauge_id.values),
        'lead_time': (['lead_time'], range(10)),
    },
  )
  
  # Convert m3/s (cms) to mm/day.
  normalized_discharge = normalize_observation(
    unnormalized_discharge=grdc_observation_data[metrics_utils.UNNORMALIZED_OBS_VARIABLE],
#     drain_area=area.to_series()
  )

#   # Remove unused dimensions.
#   grdc_observation_data = grdc_observation_data.drop(
#     [
#         'country',
#         'geo_x',
#         'geo_y',
#         'geo_z',
#         'owneroforiginaldata',
#         'river_name',
#         'station_name',
#         'timezone',
#     ]
#   )

  return xarray.merge([normalized_discharge, grdc_observation_data])
