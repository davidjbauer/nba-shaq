import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    recall_score,
    accuracy_score,
    precision_score,
    precision_recall_curve,
    f1_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def mvp_predict():
    """ Returns: mvp_summary - pandas dataframe containing players with the
                               top MVP probabilities
    """
    player_df = pd.read_json("data/1980-2018-per100-labeled.json")
    player_df19 = pd.read_json("data/2019-per-100.json")
    combined_df = pd.read_json("data/combined-player-team-stats.json")
    combined_df19 = pd.read_json("data/2019-combined.json")
    y_features = "MVP"
    x_features = [
        "Player",
        "Year",
        "Tm",
        "MP",
        "FG",
        "FGA_x",
        "3P",
        "3PA",
        "2P",
        "2PA",
        "FT",
        "FTA_x",
        "ORB",
        "DRB",
        "TRB",
        "AST_x",
        "STL_x",
        "BLK_x",
        "TOV_x",
        "PF_x",
        "PTS_x",
        "ORtg",
        "DRtg",
        "WIN_PCT",
        "CONF_RANK",
        "USG%",
        "WS",
    ]
    x_all = combined_df[x_features]
    y_all = combined_df[y_features]
    # Drop the columns containing text data, leaving only numerical data
    drop_txt = lambda x: x.drop(columns=["Player", "Year", "Tm"])
    x_all_num = drop_txt(x_all)

    x_old = x_all.loc[x_all.Year > 1980]
    y_old = y_all.loc[x_all.Year > 1980]
    x_19 = combined_df19[x_features]
    x_train, x_test, y_train, y_test = train_test_split(x_old, y_old, test_size=0.3)

    x_train_num = drop_txt(x_train)
    x_test_num = drop_txt(x_test)
    x_old_num = drop_txt(x_old)
    x_new_num = drop_txt(x_19)
    testing = False
    show_pr_curves = testing
    if not testing:
        x_train_num = x_old_num
        y_train = y_old

    scale_pipe = lambda x: Pipeline([("scaler", StandardScaler()), x])
    calibrate = lambda x: CalibratedClassifierCV(x, method="isotonic", cv=5)

    xgb_calib = calibrate(XGBClassifier(learning_rate=0.1, n_jobs=2, n_estimators=100))
    xgb_clf = scale_pipe(("xgb", xgb_calib))
    xgb_clf.fit(x_train_num, y_train)
    if show_pr_curves:
        precf_gb, recf_gb, thrf_gb = precision_recall_curve(
            y_test, xgb_clf.predict_proba(x_test_num).transpose()[1], pos_label=1
        )
        plt.plot(recf_gb, precf_gb)

    xgb_pred_prob = xgb_clf.predict_proba(x_new_num).transpose()[1]
    xgb_pred_indices = np.where(xgb_pred_prob > 0.1)

    rf_calib = calibrate(
        RandomForestClassifier(n_jobs=2, n_estimators=300, max_features="auto")
    )
    rf_clf = scale_pipe(("forest", rf_calib))
    rf_clf.fit(x_train_num, y_train)
    rf_pred_prob = rf_clf.predict_proba(x_new_num).transpose()[1]
    if show_pr_curves:
        precf_rf, recf_rf, thrf_rf = precision_recall_curve(
            y_test, rf_clf.predict_proba(x_test_num).transpose()[1], pos_label=1
        )
        plt.plot(recf_rf, precf_rf)

    mlp = calibrate(
        MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100, 100, 100, 100)
        )
    )
    mlp_clf = scale_pipe(("mlp", mlp))
    mlp_clf.fit(x_train_num, y_train)
    mlp_pred_prob = mlp_clf.predict_proba(x_new_num).transpose()[1]
    if show_pr_curves:
        precf_rf, recf_rf, thrf_rf = precision_recall_curve(
            y_test, mlp_clf.predict_proba(x_test_num).transpose()[1], pos_label=1
        )
        plt.plot(recf_rf, precf_rf)

    stacked_prob = np.mean(
        [1 * xgb_pred_prob, 1 * rf_pred_prob, 2 * mlp_pred_prob], axis=0
    )
    stacked_indices = np.where(stacked_prob > 0.01)
    sum_prob = stacked_prob.sum()
    x_19["Probability"] = stacked_prob / sum_prob
    mvp_candiates = x_19.iloc[stacked_indices]
    mvp_candidates = x_19.loc[x_19.Probability > 0.01]
    mvp_candidates.to_json("data/mvp-prediction-2019.json")

    mvp_summary = mvp_candiates.sort_values(by="Probability", ascending=False)[
        ["Player", "Probability"]
    ]
    return mvp_summary
