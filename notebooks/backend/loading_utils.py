# """Utilities for loading files for global model experiments."""

import os
import pathlib
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


def create_remote_folder_if_necessary(
    path_to_create: pathlib.Path
) -> None:
    """Creates a remote directory."""
    if os.path.exists(path_to_create):
        return
    os.makedirs(path_to_create)

# -- Utilities for loading metadata files --------------------------------------


def load_hydroatlas_country_file() -> pd.DataFrame:
    with open(data_paths.HYDROATLAS_COUNTRIES_PATH, 'rt') as f:
        hydroatlas_countries = pd.read_csv(f)
    hydroatlas_countries['HyBAS ID'] = hydroatlas_countries[
        'HyBAS ID'].apply(lambda x: int(x.split('_')[1]))
    hydroatlas_countries.set_index('HyBAS ID', inplace=True)
    return hydroatlas_countries


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


def load_full_hydroatlas_attributes_file() -> pd.DataFrame:

    with open(data_paths.FULL_HYDROATLAS_ATTRIBUTES_FILENAME, 'rt') as f:
        hybas_attributes = pd.read_csv(f, index_col='Unnamed: 0')

    column_mapper = {col: col.split('HYDRO_ATLAS:')[1] for col in hybas_attributes.columns}
    hybas_attributes = hybas_attributes.rename(columns=column_mapper)

    return hybas_attributes


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


def load_glofas_v4_calibration_metadata_file() -> pd.DataFrame:
    """Loads the GloFAS metadata file."""
    with open(data_paths.GLOFAS_v4_GAUGED_METADATA_FILENAME, 'r') as f:
        metadata = pd.read_csv(f)
    metadata['grdc_number'] = metadata['grdc_number'].apply(lambda x: f'GRDC_{x}')
    metadata.set_index('grdc_number', inplace=True)
    return metadata


def load_glofas_v4_metadata_file() -> pd.DataFrame:
    with open(data_paths.GLOFAS_METADATA_FILENAME, 'rt') as f:
        df = pd.read_csv(f, index_col='station_id_num_GRDC')
        df.index = [f'GRDC_{idx}' for idx in df.index]
    return df
   

def load_grdc_record_length_file() -> pd.DataFrame:
    with open(data_paths.GRDC_RECORD_LENGTH_FILENAME, 'rt') as f:
        record_lengths_df = pd.read_csv(f, index_col='Gauge ID')
    return record_lengths_df


def load_gauge_country_file() -> pd.DataFrame:
    with open(data_paths.GAUGE_COUNTRY_FILE, 'rt') as f:
        gauge_countries = pd.read_csv(f, index_col='gauge_id')
    return gauge_countries


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

# --- Utilities for Loading Extracted Google Model Runs ------------------------


def load_google_model_for_one_gauge(
    experiment: str,
    gauge: str,
) -> xarray.Dataset:
    """Loads a whole Google model result from one gauge."""
    results_file = data_paths.GOOGLE_MODEL_RUNS_DIR / experiment / f'{gauge}.nc'
    return _load_gauge(results_file=results_file)


def load_ungauged_experiment_model_runs(
    experiment: str,
    gauges: list[str],
) -> xarray.Dataset:
    """Loads all runs for one ungauged experiment."""

    print(f'Working on experiment: {experiment}')

    # Loads model runs for all gauges in an experiment.
    results_dict = {}
    for gauge in tqdm.tqdm(gauges):
        results_dict[gauge] = load_google_model_for_one_gauge(
            experiment=experiment,
            gauge=gauge,
        )

    # Concats into a single xarray dataset indexed by gauge.
    return xarray.concat(
        [ds for ds in results_dict.values() if ds is not None],
        dim='gauge_id'
    )


def load_all_experimental_model_runs(
    gauges: list[str],
    experiments: list[str] = data_paths.EXPERIMENTS
) -> dict[str, xarray.Dataset]:
    """Loads all runs for all Google model experiments."""

    return {
        experiment: load_ungauged_experiment_model_runs(
            experiment=experiment,
            gauges=gauges,
        ) for experiment in experiments
    }


# --- Utilities for Loading Extracted GloFAS Model Runs ------------------------

def load_glofas_reanalysis_for_one_gauge(
    gauge: str,
    time_period: Optional[list[str]] = None,
    lead_times: Optional[list[int]] = None,
) -> xarray.Dataset:
    """Loads a timeselice of GloFAS model results from one gauge."""
    results_file = data_paths.GLOFAS_MODEL_RUNS_DIR / f'{gauge}.nc'
    return _load_gauge(
        results_file=results_file,
        time_period=time_period,
        lead_times=lead_times,
    )


def load_glofas_model_runs(
    gauges: list[str],
) -> xarray.Dataset:
    """Loads all runs for one ungauged experiment."""

    data = xarray.load_dataset(data_paths.GLOFAS_MODEL_RUNS)

    gauge_ids = [f'GRDC_{str(int(gauge)).zfill(7)}' for gauge in data.statid.values]
    gauge_ids_da = xarray.DataArray(
        gauge_ids,
        coords={'station': data.station.values},
        dims=['station']
    )
    data['statid'] = gauge_ids_da
    data = data.swap_dims({'station': 'statid'})
    data = data.drop('station')
    data = data.rename({'statid': 'gauge_id'})    
                
    data = data.drop(
        [
            'statlon',
            'statlat',
            'statups',
            'mappedlon',
            'mappedlat',
            'mappedups',
            'row',
            'col',
            'row1',
            'col1'
        ]
    )
    data = data.rename({'dis': metrics_utils.GLOFAS_VARIABLE})
        
    existing_gauges = [gauge for gauge in gauges if gauge in data.gauge_id.values]
    return data.sel({'gauge_id': existing_gauges})


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
    )

    return xarray.merge([normalized_discharge, grdc_observation_data])
