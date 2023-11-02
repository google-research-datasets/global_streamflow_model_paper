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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support
import tqdm
from typing import Optional

from backend import evaluation_utils

N_KFOLD = 25
N_TREES = 100


def make_predictors(
    x: pd.DataFrame,
    y: pd.DataFrame,
    metric: str,
    bins: Optional[list[int]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if bins is not None:
        bin_names = [idx for idx, bin in enumerate(bins[1:])]
        y_cat = pd.cut(y, bins=bins, include_lowest=True, labels=bin_names)
    else:
        y_cat = y

    joined = pd.concat([x, y_cat], axis=1).dropna(axis=0)
    joined.replace([np.inf, -np.inf], np.nan, inplace=True)
    joined = joined.dropna()
    predictand = joined[metric]
    predictors = joined.drop(metric, axis=1)

    print(f'There are {predictors.shape[1]} predictors.')
    print(f'There are {predictors.shape[0]} samples.')

    return predictors, predictand


def train_kfold(
    predictors: pd.DataFrame,
    predictand: pd.DataFrame,
    classifier: bool = False,
    regression: bool = False
):

    if regression and classifier:
        raise ValueError('Can only use regression *or* classifier, got both.')
    if not regression and not classifier:
        raise ValueError('Must use either regression or classifier.')

    # Initialize storage.
    basins = list(predictand.index)
    y_hat = pd.Series(index=basins)

    # Create a separate split for each ensemble member.
    if classifier:
        kf = model_selection.StratifiedKFold(
            n_splits=N_KFOLD,
            random_state=None,
            shuffle=True
        )
        splits = kf.split(basins, predictand)
    elif regression:
        kf = model_selection.KFold(
            n_splits=N_KFOLD,
            random_state=None,
            shuffle=True
        )
        splits = kf.split(basins)

    # Train and test the model.
    for kfold, (train_index, test_index) in enumerate(tqdm.tqdm(splits)):
        print(f'Training fold # {kfold}')

        # Train/test split.
        train_basins = [basins[idx] for idx in train_index]
        test_basins = [basins[idx] for idx in test_index]
        train_y = predictand.loc[train_basins]
        train_x = predictors.loc[train_y.index]
        test_x = predictors.loc[test_basins]

        # RF model.
        if regression:
            rf = RandomForestRegressor(n_estimators=N_TREES, random_state=42)
        elif classifier:
            rf = RandomForestClassifier(n_estimators=N_TREES, random_state=42, class_weight='balanced')
        rf.fit(train_x, train_y)

        # Predictions.
        y_hat.loc[test_basins] = rf.predict(test_x)

    return y_hat


def feature_importances(
    predictors: pd.DataFrame,
    predictand: pd.DataFrame,
    classifier: bool = False,
    regression: bool = False
):

    if regression and classifier:
        raise ValueError('Can only use regression *or* classifier, got both.')
    if not regression and not classifier:
        raise ValueError('Must use either regression or classifier.')

    # Train a predictor with all of the data.
    if regression:
        rf = RandomForestRegressor(n_estimators=N_TREES, random_state=42)
    elif classifier:
        rf = RandomForestClassifier(n_estimators=N_TREES, random_state=42)
    rf.fit(predictors, predictand)

    # Extract importances.
    importances = pd.Series(
        index=predictors.columns, data=rf.feature_importances_
    ).sort_values(ascending=False)
    importance_stds = np.std(
        [tree.feature_importances_ for tree in rf.estimators_], axis=0
    )

    return importances, importance_stds


def plot_feature_importances(
    importances: pd.DataFrame,
    ax: plt.Axes
):
    importances = importances.sort_values(ascending=False)
    ax.bar(importances.index, importances.values)
    ax.set_title(
        'Feature Importances',
        fontsize=evaluation_utils.NATURE_FONT_SIZES['title']
    )
    ax.set_ylabel(
        'Mean Decrease in Impurity', 
        fontsize=evaluation_utils.NATURE_FONT_SIZES['axis_label']
    )
    ax.grid(c='#EEE')
    ax.set_xticks(
        range(len(importances.index)),
        importances.index,
        rotation=60,
        ha='right',
        fontsize=evaluation_utils.NATURE_FONT_SIZES['tick_labels']
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=evaluation_utils.NATURE_FONT_SIZES['tick_labels']
    )
    ax.set_xlabel(None)
    ax.set_ylabel(
        ax.get_ylabel(), 
        fontsize=evaluation_utils.NATURE_FONT_SIZES['axis_label']
    )
    ax.set_axisbelow(True)
