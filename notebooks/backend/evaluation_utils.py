"""Utiltities for producing plots & statis in the Global Modleing Paper."""

import os
import pathlib
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import tqdm

from backend import data_paths
from backend import loading_utils
from backend import metrics_utils


# Which return periods to evalaute.
RETURN_PERIODS = [1.01, 2, 5, 10, 20, 50]

# Country outlines. Read a GeoDataFrame from file (this is the only way I've
# found within the Google environment).
COUNTRIES = gpd.read_file(
    open(data_paths.COUNTRY_POLYGONS_FILENAME))
COUNTRIES.rename(columns={'name': 'country_name'}, inplace=True)
COUNTRIES = COUNTRIES[COUNTRIES['continent'] != 'Antarctica']

# Parameters for choosing realistic map aspect ratio.
SPHERE_RATIO = 2.05458
LONGITUDE_FIG_SIZE = 16
LATITIDE_FIG_SIZE = LONGITUDE_FIG_SIZE / SPHERE_RATIO

EXPERIMENT_NAMES = {
    name: name.replace('_', ' ').title()
    for name in data_paths.EXPERIMENTS if name != 'full_run'
}
EXPERIMENT_NAMES['full_run'] = 'Gauged Basins Run'
EXPERIMENT_NAMES['kfold_splits'] = 'AI Model'
EXPERIMENT_NAMES['glofas_reanalysis'] = 'GloFAS'
# EXPERIMENT_NAMES['glofas_reforecasts'] = 'GloFAS Reforecasts'

METRIC_NAMES = {
    'NSE': 'Nash Sutcliffe Efficiency (NSE)',
    'log-NSE': 'NSE of Log-Transformed Values',
    'Alpha-NSE': 'Standard Deviation Ratio',
    'Beta-NSE': 'Bias',

    'KGE': 'Kling-Gupta Efficiency (KGE)',
    'log-KGE': 'KGE of Log-Transformed Values',
    'Pearson-r': 'Correlation Coefficient',
    'Beta-KGE': 'Bias Ratio',
}

METRICS_AXIS_LIMITS = {
    'KGE': (-1, 1),
    'log-KGE': (-1, 1),
    'Pearson-r': (0, 1),
    'Beta-KGE': (0, 2),

    'NSE': (-1, 1),
    'log-NSE': (-1, 1),
    'Alpha-NSE': (0, 2),
    'Beta-NSE': (-1, 1),
}

prop_cycle = plt.rcParams['axes.prop_cycle']
COLORS = prop_cycle.by_key()['color']

LEAD_TIME_LINESTYLES = ['-'] + [':']*8
LEAD_TIME_ALPHAS = [0.8**lt for lt in data_paths.LEAD_TIMES]

EXPERIMENT_LINESTYLES = ['-', '--']
EXPERIMENT_COLORS = {
    experiment: color 
    for experiment, color in zip(list(EXPERIMENT_NAMES.keys())[1:], COLORS[2:])
}
# Switch the order of the first two colors due to sns boxplot.
# Also, I think it looks better this way.
EXPERIMENT_COLORS[loading_utils.GLOFAS_REANALYSIS_VARIABLE_NAME] = COLORS[1]
EXPERIMENT_COLORS['kfold_splits'] = COLORS[0]


def f1_from_precision_and_recall_dfs(
    precision_df: pd.DataFrame,
    recall_df: pd.DataFrame
) -> pd.DataFrame:
  f1 = 2*(precision_df * recall_df / (precision_df + recall_df))
  f1[(precision_df + recall_df) == 0] = 0
  return f1


def load_return_period_metrics(
    base_path: pathlib.Path,
    experiments: list[str],
    gauges: list[str],
    metric: str,
) -> dict[str, dict[str, pd.DataFrame]]:
  """Load and concatenate return period metrics."""

  if metric.lower() not in ['precision', 'recall']:
    raise ValueError('Can only load precisions or recalls.')

  scores_by_lead_time = {
      experiment: {
          lead_time: pd.DataFrame(
              index=gauges,
              columns=RETURN_PERIODS,
              dtype=float
          ) for lead_time in data_paths.LEAD_TIMES
      } for experiment in experiments
  }

  for experiment in experiments:
    print(f'Working on experiment {experiment} ...')

    for gauge in tqdm.tqdm(gauges):

      filepath = metrics_utils.save_metrics_df(
          df=None,
          metric=gauge,
          base_path=base_path,
          path_modifier=f'{experiment}/{metric.lower()}',
      )
    
      if os.path.exists(filepath):
        with open(filepath, 'r') as f:
          scores = pd.read_csv(f, index_col=[0, 1])
        idx = pd.IndexSlice
        for lead_time in data_paths.LEAD_TIMES:
          scores_by_lead_time[experiment][lead_time].loc[gauge] = (
              scores.loc[idx[:, 2], str(lead_time)].values
          )
      else:
        for lead_time in data_paths.LEAD_TIMES:
          scores_by_lead_time[experiment][lead_time].loc[gauge] = np.nan

  return scores_by_lead_time


def score_by_country(
    scores: pd.DataFrame,
    gauge_to_country_mapping: pd.DataFrame,
    countries: list[str],
    gauges: list[str],
    return_period: int,
    means: bool = False,
    medians: bool = False
) -> pd.DataFrame:
  """Create dataframes of reliability scores by country."""

  if means and medians:
    raise ValueError(
        'Cannot calculate both means and medians at the same time.')
  if not means and not medians:
    raise ValueError('Must specify whether to calculate means or medians.')

  gauge_to_country_mapping = gauge_to_country_mapping.loc[gauges]
  experiments = list(scores.keys())

  scores_by_country = {
      experiment:
      pd.DataFrame(index=countries, columns=data_paths.LEAD_TIMES, dtype=float)
      for experiment in experiments
  }

  for experiment in experiments:
    print(f'Working on experiment: {experiment} ...')

    for country in tqdm.tqdm(countries):
      gauges_for_country = [
          idx
          for idx in gauge_to_country_mapping.index
          if gauge_to_country_mapping.loc[idx]['Country'] == country
      ]

      for lead_time in data_paths.LEAD_TIMES:
        if means:
          scores_by_country[experiment].loc[country, lead_time] = (
              scores[experiment][lead_time]
              .loc[gauges_for_country, return_period]
              .mean()
          )
        if medians:
          scores_by_country[experiment].loc[country, lead_time] = (
              scores[experiment][lead_time]
              .loc[gauges_for_country, return_period]
              .median()
          )

  return scores_by_country

# --- Plotting Functions -------------------------------------------------------


def save_figure(filename: str):
  with open(data_paths.FIGURES_DIR / filename, 'wb') as f:
    plt.savefig(f, dpi=600)


def plot_cdf(
    scores: np.ndarray,
    ax: plt.Axes,
    xlabel: str,
    label: Optional[str] = None,
    xlim: Optional[list[float]] = None,
    lw: float = 2,
    color: str = None,
    ls: str = '-',
    alpha: float = 1
):
  """Plots a CDF for a given series of data."""

  # Make the CDF
  x_data, y_data = _empirical_cdf(x=scores)

  # Plot the CDF
  if color is not None:
    if label is not None:
      ax.plot(x_data, y_data, label=label, lw=lw, c=color, ls=ls, alpha=alpha)
    else:
      ax.plot(x_data, y_data, lw=lw, c=color, ls=ls, alpha=alpha)
  else:
    if label is not None:
      ax.plot(x_data, y_data, label=label, lw=lw, ls=ls, alpha=alpha)
    else:
      ax.plot(x_data, y_data, lw=lw, ls=ls, alpha=alpha)

  # Aesthetics
  ax.legend()
#   plt.legend(loc='upper left')
  ax.set_ylabel('Fraction of Gauges')
  ax.set_xlabel(xlabel)

  if xlim is not None:
    ax.set_xlim(xlim)
  _remove_spines_from_axis(ax)


def _empirical_cdf(x: np.ndarray):
  """Calculates empirical cumulative density function."""
  xs = np.sort(np.squeeze(x))
  ys = np.array(range(len(x))) / len(x)
  return xs, ys


def _remove_spines_from_axis(ax: plt.Axes):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)


def spatial_plot(
    metric_data: pd.DataFrame,
    latlon: pd.DataFrame,
    metric: str,
    ms: int = 10,
    experiment: Optional[str] = None,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
):
  """Plot a well-formatted global figure for some metric."""

  plotdata = pd.concat([metric_data, latlon], axis=1).dropna()

  if ax is None:
    ax = COUNTRIES.boundary.plot(color='gainsboro', linewidth=0.5)
    fig = ax.get_figure()
    fig.set_size_inches(LONGITUDE_FIG_SIZE, LATITIDE_FIG_SIZE)
  else:
    ax = COUNTRIES.boundary.plot(color='gainsboro', linewidth=0.5, ax=ax)

  if vmin is None:
    vmin = METRICS_AXIS_LIMITS[metric][0]
  if vmax is None:
    vmax = METRICS_AXIS_LIMITS[metric][1]

  points = ax.scatter(
      plotdata['longitude'], plotdata['latitude'],
      c=plotdata[metric],
      s=ms,
      vmin=vmin,
      vmax=vmax,
      cmap=cmap
  )

  if title is not None:
    ax.set_title(title, fontsize=16)
  elif experiment is not None:
    ax.set_title(f'{EXPERIMENT_NAMES[experiment]} ({metric})', fontsize=20)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.set_xticks([])
  ax.set_yticks([])

  if colorbar:
    cbaxes = inset_axes(ax, width='1%', height='90%', loc='center right')
    cbar = plt.colorbar(points, cax=cbaxes, orientation='vertical')
    cbar.solids.set_edgecolor('face')
    cbar.set_label(metric, rotation=90, fontsize=14)

  return ax


def plot_gdp_gauge_record_correlation(
    most_recent_gdp:pd.Series,
    record_lengths_by_country: pd.Series,
    all_labels: bool = True
):

  # Align data along index of two pandas series.
  df = pd.concat([most_recent_gdp, record_lengths_by_country], axis=1)

  # Clean and transform data for plotting.
  df['Most Recent GDP'] = df['Most Recent GDP'].apply(
      lambda x: np.log(x)).values
  df['Record Length'] = df['Record Length'].apply(
      lambda x: np.log(x)).values
  df = df.replace([-np.inf], np.nan).dropna()

  # Extract data for plotting.
  x = df['Most Recent GDP'].values
  y = df['Record Length'].values

  # Plot stuff.
  _, ax = plt.subplots(1, 1, figsize=(8, 6))
  plt.scatter(x, y)

  # Best fit line.
  a, b = np.polyfit(x, y, 1)
  plt.plot([min(x), max(x)], [a*min(x)+b, a*max(x)+b], 'k:')

  # Aesthetics.
  plt.xlabel('(log) GDP in USD', fontsize=16)
  plt.ylabel('(log) Total Years of Record in Country', fontsize=16)
  ax = plt.gca()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  # Add country labels.
  countries_to_label = [
      'United States',
      'Canada',
      'Brazil',
      'China',
      'Germany',
      'France',
      'South Africa',
      'Japan',
      'Russia',
      'Mexico',
      'Sweeden',
      'Switzerland',
      'South Korea',
      'Italy',
      'Nigeria',
      'Bangladesh',
      'Lesotho',
      'Namibia',
      'Ecuador',
      'Jamaica',
      'Liberia',
      'Bulgaria',
      'Togo',
      'Moldova',
      'Gambodia',
      'São Tomé and Príncipe',
      'Mauritania',
      'Hondouras',
      'Sierra Leone',
      'Zambia',
      'Honduras',
      'Sweden',
      'Indonesia',
      'Pakistan',
      'Azerbaijan',
      'Guinea',
      'Columbia',
      'Georgia',
      'Guyana',
      'Kyrgystan',
      'Zimbabwe',
      'Slovakia',
      'Romania',
      'Chile',
      'Uruguay',
      'Oman',
      'Costa Rica',
      'Belarus',
      'Venezuela',
      'Colombia',
      'Sri Lanka',
      'Israel',
      'Kazakhstan',
  ]
  if all_labels:
    countries_to_label = df.index
    countries_to_label += [a for a in df.index if np.random.rand() > 0.5]

  for country, data in df.dropna().iterrows():
    if country in countries_to_label:
      plt.text(
          x=data['Most Recent GDP'],
          y=data['Record Length'],
          s=f' {country}',
          fontdict=dict(color='indigo', size=10),
      )

  # # Add correlation coefficient to plot.
  # correlation = np.corrcoef(x, y)[0, 1]
  # plt.text(
  #     x=6,
  #     y=10.5,
  #     s=f'r = {correlation:0.3}',
  #     fontdict=dict(color='k', size=15),
  #     bbox=dict(facecolor='w', alpha=1),
  # )


def hydrograph_metrics_cdf_plots(
    title: str,
    time_period_identifier: str,
    lead_times: list[str]
):
  """Plot a set of hydrograph metric CDFs."""

  _, axes = plt. subplots(2, 4, figsize=(12, 6))

  for ax, metric in zip(axes.flatten(), METRICS_AXIS_LIMITS):

    metric_filename =  f'{metric}.csv'

    # Load Google and GloFAS metrics.
    glofas_path = data_paths.METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'glofas_v3' / time_period_identifier / 'glofas_reanalysis' / metric_filename
    glofas_metrics_data = metrics_utils.load_metrics_df(filepath=glofas_path)

    google_basepath = data_paths.METRICS_DIR / 'hydrograph_metrics' / 'per_metric' / 'google' / time_period_identifier / data_paths.GOOGLE_MODEL
    google_metrics_data = {}
    for experiment in data_paths.EXPERIMENTS:
        google_metrics_data[experiment] = metrics_utils.load_metrics_df(
            filepath=google_basepath / experiment / metric_filename)

    for lead_time in lead_times:
        plotdata = pd.concat(
            [
                google_metrics_data[experiment][str(lead_time)].rename(experiment)
                for experiment in data_paths.EXPERIMENTS
            ],
            axis=1
        )
        if lead_time == 0:
            plotdata = pd.concat([plotdata, glofas_metrics_data[
                str(lead_time)].rename('glofas_reanalysis')], axis=1)

        plotdata.dropna(inplace=True)

        for experiment in plotdata:

            label = None
            if lead_time == 0:
                label = EXPERIMENT_NAMES[experiment]

            if len(plotdata[experiment]) > 0:
                plot_cdf(
                    scores=plotdata[experiment].values, 
                    label=label,
                    xlabel=METRIC_NAMES[metric],
                    xlim=METRICS_AXIS_LIMITS[metric],
                    ax=ax,
                    color=EXPERIMENT_COLORS[experiment],
                    ls=LEAD_TIME_LINESTYLES[lead_time],
                    alpha=LEAD_TIME_ALPHAS[lead_time]
                )

    ax.grid(c='#EEE')
    if ax is not axes.flatten()[0] and ax.get_legend() is not None:
      ax.get_legend().remove()

    ax.set_title(title)

  plt.tight_layout()


def return_period_cdf_plots(
  precisions: dict[str, dict[str, pd.Series]],
  recalls: dict[str, dict[str, pd.Series]],
  title: str
):
  _, axes = plt. subplots(2, 4, figsize=(16, 8))
  axes = axes.transpose()

  for idx, return_period in enumerate(
      RETURN_PERIODS[:-2]):

    for experiment in EXPERIMENT_NAMES:

      if experiment not in precisions:
        continue

      if experiment in data_paths.EXPERIMENTS:
        lead_times = data_paths.LEAD_TIMES
        basepath = pathlib.Path(
            '/home/gsnearing/data/metrics/hydrograph_metrics/per_metric/google/1980/dual_lstm')
      else:
        lead_times = [0]
        basepath = pathlib.Path(
            '/home/gsnearing/data/metrics/hydrograph_metrics/per_metric/glofas_v3/1980/dual_lstm')

      for lead_time in lead_times:

        label = None
        if lead_time == 0:
          label = EXPERIMENT_NAMES[experiment]

        plot_cdf(
            scores=precisions[
                experiment][lead_time][return_period].dropna().values, 
            label=label,
            xlabel=f'{int(return_period)}-Year Return Period Precision',
            xlim=[0, 1],
            ax=axes[idx, 0],
            color=EXPERIMENT_COLORS[experiment],
            ls=LEAD_TIME_LINESTYLES[lead_time],
            alpha=LEAD_TIME_ALPHAS[lead_time]
        )

        plot_cdf(
            scores=recalls[
                experiment][lead_time][return_period].dropna().values, 
            label=label,
            xlabel=f'{int(return_period)}-Year Return Period Recall',
            xlim=[0, 1],
            ax=axes[idx, 1],
            color=EXPERIMENT_COLORS[experiment],
            ls=LEAD_TIME_LINESTYLES[lead_time],
            alpha=LEAD_TIME_ALPHAS[lead_time]
        )

    axes[idx, 0].grid(c='#EEE')
    axes[idx, 1].grid(c='#EEE')
    axes[idx, 0].set_title(title)
    axes[idx, 1].set_title(title)

    axes[idx, 1].get_legend().remove()
    if idx != 0:
      axes[idx, 0].get_legend().remove()

  plt.tight_layout()
