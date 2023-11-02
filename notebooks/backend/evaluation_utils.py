"""Utiltities for producing plots & statis in the Global Modleing Paper."""

import os
import pathlib
from typing import Optional, Tuple

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

EXPERIMENT_NAMES = {
    name: name.replace('_', ' ').title()
    for name in data_paths.EXPERIMENTS if name != 'full_run'
}
EXPERIMENT_NAMES['full_run'] = 'Gauged Basins Run'
EXPERIMENT_NAMES['hydrologically_separated'] = 'Hydrologically\nSeparated'
EXPERIMENT_NAMES['kfold_splits'] = 'AI Model'
EXPERIMENT_NAMES[metrics_utils.GLOFAS_VARIABLE] = 'GloFAS'
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
    'NSE': (-1, 1),
    'log-NSE': (-1, 1),
    'Alpha-NSE': (0, 2),
    'Beta-NSE': (-1, 1),
    'KGE': (-1, 1),
    'log-KGE': (-1, 1),
    'Beta-KGE': (0, 2),
    'Pearson-r': (0, 1),
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
EXPERIMENT_COLORS[metrics_utils.GLOFAS_VARIABLE] = COLORS[1]
EXPERIMENT_COLORS['kfold_splits'] = COLORS[0]

MM2IN = 25.4
NATURE_FIG_SIZES = {
    'one_column': 89 / MM2IN,
    'one_half_column': 120 / MM2IN,
    'two_column': 183 / MM2IN,
}

NATURE_FONT_SIZES = {
    'title': 8,
    'axis_label': 8,
    'text_labels': 5,
    'tick_labels': 6,
    'legend': 6,
    'markersize': 0.02
}

# NATURE_FONT_SIZES['one_half_column'] = NATURE_FONT_SIZES['one_column'].copy()
# for key, val in NATURE_FONT_SIZES['one_column'].items():
#     NATURE_FONT_SIZES['one_half_column'][key] = val * NATURE_FIG_SIZES['one_half_column'] / NATURE_FIG_SIZES['one_column']

# NATURE_FONT_SIZES['two_column'] = NATURE_FONT_SIZES['one_column'].copy()
# for key, val in NATURE_FONT_SIZES['one_column'].items():
#     NATURE_FONT_SIZES['two_column'][key] = val * NATURE_FIG_SIZES['two_column'] / NATURE_FIG_SIZES['one_column']

# Parameters for choosing realistic map aspect ratio.
SPHERE_RATIO = 2.05458
LATITIDE_FIG_SCALER = 1 / SPHERE_RATIO


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

# --- Plotting Functions -------------------------------------------------------

def save_figure(filename: str):
    with open(data_paths.FIGURES_DIR / f'{filename}.eps', 'wb') as f:
        plt.savefig(f, dpi=600, format='eps')
    with open(data_paths.FIGURES_DIR / f'{filename}.png', 'wb') as f:
        plt.savefig(f, dpi=300, transparent=True)


def plot_cdf(
    scores: np.ndarray,
    ax: plt.Axes,
    xlabel: str,
    label: Optional[str] = None,
    xlim: Optional[list[float]] = None,
    lw: float = 2,
    color: str = None,
    ls: str = '-',
    alpha: float = 1,
    legend: bool = True
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
    if legend:
        ax.legend()
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
    ms: Optional[float] = None,
    experiment: Optional[str] = None,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
    figure_size: str = 'one_half_column'
):
    """Plot a well-formatted global figure for some metric."""

    plotdata = pd.concat([metric_data, latlon], axis=1).dropna()

    if ax is None:
        ax = COUNTRIES.boundary.plot(color='gainsboro', linewidth=0.2)
        fig = ax.get_figure()
        fig.set_size_inches(
            NATURE_FIG_SIZES[figure_size],
            LATITIDE_FIG_SCALER * NATURE_FIG_SIZES[figure_size]
        )
    else:
        ax = COUNTRIES.boundary.plot(color='gainsboro', linewidth=0.2, ax=ax)

    if vmin is None:
        vmin = METRICS_AXIS_LIMITS[metric][0]
    if vmax is None:
        vmax = METRICS_AXIS_LIMITS[metric][1]

    if ms is None:
        ms = NATURE_FONT_SIZES['markersize']
        
    points = ax.scatter(
      plotdata['longitude'], plotdata['latitude'],
      c=plotdata[metric],
      s=ms,
      vmin=vmin,
      vmax=vmax,
      cmap=cmap
    )

    if title is not None:
        ax.set_title(
            title, 
            fontsize=NATURE_FONT_SIZES['title']
        )
    elif experiment is not None:
        ax.set_title(
            f'{EXPERIMENT_NAMES[experiment]} ({metric})',
            fontsize=NATURE_FONT_SIZES['title']
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = None
    if colorbar:
        cbaxes = inset_axes(ax, width='1%', height='90%', loc='center right')
        cbar = plt.colorbar(points, cax=cbaxes, orientation='vertical')
        cbar.solids.set_edgecolor('face')
        cbar.set_label(None)
#         cbar.set_label(
#             metric, 
#             rotation=90, 
#             fontsize=NATURE_FONT_SIZES['axis_label']
#         )
        cbar.ax.tick_params(labelsize=NATURE_FONT_SIZES['tick_labels']) 

    return ax, cbar


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

    # Initialize plot.
    _, ax = plt.subplots(
        1, 1, 
        figsize=(
            NATURE_FIG_SIZES['one_column'], 
            NATURE_FIG_SIZES['one_column']*3/4,
        )
    )

    
    # Best fit line.
    a, b = np.polyfit(x, y, 1)
    plt.plot(
        [min(x), max(x)], 
        [a*min(x)+b, a*max(x)+b], 
        ':',
        c='gray'
    )

    # Plot stuff.
    plt.scatter(
        x, 
        y,
        s=3,
    )

    # Aesthetics.
    plt.xlabel(
        '(log) GDP in USD', 
        fontsize=NATURE_FONT_SIZES['axis_label']
    )
    plt.ylabel(
        '(log) Total Years of Record in Country', 
        fontsize=NATURE_FONT_SIZES['axis_label']
    )
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        fontsize=NATURE_FONT_SIZES['tick_labels']
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=NATURE_FONT_SIZES['tick_labels']
    )

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
#       'Russia',
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
#       'Indonesia',
      'Pakistan',
      'Azerbaijan',
      'Guinea',
      'Georgia',
#       'Guyana',
      'Kyrgystan',
      'Zimbabwe',
      'Slovakia',
      'Romania',
      'Chile',
#       'Uruguay',
      'Oman',
#       'Costa Rica',
      'Belarus',
      'Venezuela',
#       'Colombia',
      'Sri Lanka',
      'Israel',
#       'Kazakhstan',
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
              fontdict=dict(
                  color='indigo', 
                  size=NATURE_FONT_SIZES['text_labels']
              ),
            )

    plt.tight_layout()

#     # Add correlation coefficient to plot.
#     correlation = np.corrcoef(x, y)[0, 1]
#     plt.text(
#         x=6,
#         y=10.5,
#         s=f'r = {correlation:0.3}',
#         fontdict=dict(color='k', size=15),
#         bbox=dict(facecolor='w', alpha=1),
#     )


def hydrograph_metrics_cdf_plots(
    title: str,
    glofas_basepath: pathlib.Path,
    google_basepath: pathlib.Path,
    lead_times: list[str],
    gauges: Optional[list[str]] = None,
):
    """Plot a set of hydrograph metric CDFs."""

    _, axes = plt. subplots(
        2, 4, 
        figsize=(
            NATURE_FIG_SIZES['one_half_column'], 
            NATURE_FIG_SIZES['one_half_column']/2,
        )
    )

    for ax, metric in zip(axes.flatten(), METRICS_AXIS_LIMITS):
        
        metric_filename =  f'{metric}.csv'

        # Load Google and GloFAS metrics.
        glofas_path = glofas_basepath / metrics_utils.GLOFAS_VARIABLE / metric_filename
        glofas_metrics_data = metrics_utils.load_metrics_df(filepath=glofas_path)
        if gauges is not None:
            glofas_metrics_data = glofas_metrics_data.loc[gauges]

        google_metrics_data = {}
        for experiment in data_paths.EXPERIMENTS:
            google_metrics_data[experiment] = metrics_utils.load_metrics_df(
                filepath=google_basepath / experiment / metric_filename)

        if gauges is not None:
            for experiment in data_paths.EXPERIMENTS:
                gauges_for_experiment = [gauge for gauge in gauges if gauge in google_metrics_data[experiment].index]
                google_metrics_data[experiment] = google_metrics_data[experiment].loc[gauges_for_experiment]

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
                    str(lead_time)].rename(metrics_utils.GLOFAS_VARIABLE)], axis=1)

            plotdata.dropna(inplace=True)

            for experiment in plotdata:

                label = None
                if lead_time == 0:
                    label = EXPERIMENT_NAMES[experiment]

                if len(plotdata[experiment]) > 0:
                    plot_cdf(
                        scores=plotdata[experiment].values, 
                        label=label,
                        xlabel=metric,
                        xlim=METRICS_AXIS_LIMITS[metric],
                        ax=ax,
                        color=EXPERIMENT_COLORS[experiment],
                        ls=LEAD_TIME_LINESTYLES[lead_time],
                        alpha=LEAD_TIME_ALPHAS[lead_time],
                        lw=0.5
                    )

        ax.grid(c='#EEE')
#         if ax is not axes.flatten()[-1] and ax.get_legend() is not None:
        ax.get_legend().remove()
            
        ax.set_ylabel(None, fontsize=0)

        ax.set_title(
            title,
            fontsize=NATURE_FONT_SIZES['title']
        )
        
        ax.set_xticks(
            np.linspace(
                METRICS_AXIS_LIMITS[metric][0], 
                METRICS_AXIS_LIMITS[metric][1], 
                5
            )
        )
        ax.set_yticks(np.linspace(0, 1, 5))

        xticklabels = list(ax.get_xticklabels())
        xticklabels[1] = None
        xticklabels[3] = None
        ax.set_xticklabels(
            xticklabels,
            fontsize=NATURE_FONT_SIZES['tick_labels']
        )
        yticklabels = list(ax.get_yticklabels())
        yticklabels[1] = None
        yticklabels[3] = None
        ax.set_yticklabels(
            yticklabels,
            fontsize=NATURE_FONT_SIZES['tick_labels']
        )
        ax.set_xlabel(
            ax.get_xlabel(), 
            fontsize=NATURE_FONT_SIZES['axis_label']
        )
        ax.set_ylabel(
            ax.get_ylabel(), 
            fontsize=NATURE_FONT_SIZES['axis_label']
        )

        
    # Plot the legend in a separate subplot.
    ax.legend(
        fontsize=NATURE_FONT_SIZES['legend']-0.5,
        loc='upper left',
        bbox_to_anchor=(-0.4, 1.0),
        handlelength=1
#         bbox_to_anchor=(1, 1)
    )
    ax.plot([-100, 100], [0.5, 0.5], c='w', lw=1000)
    ax.set_ylabel(None, fontsize=0)
    ax.grid()
#     axes[-1, -1].remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])    
    ax.set_xlabel(None)
    plt.tight_layout()


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

