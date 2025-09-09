###########################
### Importing Libraries ###
###########################

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from scipy.special import expit

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, balanced_accuracy_score)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.base import clone

from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from xgboost import XGBClassifier

from skopt import BayesSearchCV
from skopt.space import Real, Integer

from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, Callback

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from statsmodels.stats.multitest import multipletests

from itertools import product

from typing import Dict, List, Tuple

#######################################
### Defining Helper Classes for ANN ###
#######################################

# Focal Loss class for enabling training focused on the difficult to predict class
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        targets = targets.view(-1,1).type_as(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
# Class for the ANN model (binary classification)
class DeepBinary(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout_rate=0.25):
        super().__init__()
        layers = []
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim, 1))  # final logit
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Defining NeuralNet class which will be necessary for use with skorch and skopt
class NeuralNetBinaryClassifier(NeuralNetClassifier):
    def predict_proba(self, X):
        logits = self.forward(X).detach().cpu().numpy()
        probs = expit(logits)
        return np.hstack((1 - probs, probs))

#############################################################
### Defining Deep Learning Skorch Set up and search space ###
#############################################################

# Defining the base ANN model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = NeuralNetBinaryClassifier(
    module = DeepBinary,
    criterion = FocalLoss,
    criterion__alpha = 0.25,
    criterion__gamma = 2.0,
    max_epochs = 300,
    lr = 1e-3,
    optimizer = torch.optim.Adam,
    optimizer__weight_decay=1e-4,
    batch_size = 128,
    device = device,
    verbose = 0,
    callbacks=[
        EarlyStopping(
            monitor='valid_loss',
            threshold=0.01,
            patience=25,
            lower_is_better=True,
            load_best=True
        )
    ]
)

# Defining the ANN search space
deep_search_space = {
    "lr": Real(0.000001, 0.01, prior="log-uniform"),
    "module__hidden_dim": Integer(4, 256),
    "module__num_layers": Integer(1, 4),
    "module__dropout_rate": Real(0.0, 0.5),
    "criterion__alpha": Real(0.0, 0.5),
    "criterion__gamma": Real(1.0, 7.0)
}

############################################
### Defining non ANN model search spaces ###
############################################

search_spaces = {
    "SVM": (
        SVC(probability=True, class_weight="balanced", kernel = 'rbf', gamma='scale', random_state=42),
        {
            "C": Real(0.001, 1.0, prior="log-uniform")
        }
    ),
    "RandomForest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {
            "n_estimators": Integer(50, 500),
            "max_depth": Integer(2, 20),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
        }
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {
            "n_estimators": Integer(50, 500),
            "max_depth": Integer(2, 20),
            "learning_rate": Real(0.001, 0.1, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "scale_pos_weight": Real(1.0, 10.0)
        }
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
        {
            "C": Real(0.001, 1.0, prior="log-uniform")
        }
    )
}

##################################
### Defining Utility Functions ###
##################################

# Function to compute metrics
def compute_metrics(y_true, probs, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)
    return {'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, probs),
            'pr_auc': average_precision_score(y_true, probs)}

# Function for tuning thresholds with Balanced Accuracy
def tune_threshold(probs, y_true):
    thresholds = np.linspace(0.0, 1.0, 101)
    best_thr, best_score = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds,average='binary')
        if score > best_score:
            best_score, best_thr = score, t
    return best_thr, best_score

# Function to calculate pooled curves over cross-validation
def compute_pooled_curve(oof_probs_list, oof_true_list, curve_type="roc"):
    # Pool predictions and true labels across folds
    y_true = np.concatenate(oof_true_list)
    y_probs = np.concatenate(oof_probs_list)

    if curve_type == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        return fpr, tpr
    elif curve_type == "pr":
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        return recall, precision
    else:
        raise ValueError("curve_type must be 'roc' or 'pr'")

# Setting grid for cumulative AUC aggregation
def _safe_time_grid(train_times: np.ndarray,
    test_times: np.ndarray,
    n_times: int = 60,
    lower_q: float = 0.10,
    upper_q: float = 0.90) -> np.ndarray:
    """Build a per-fold evaluation grid that (1) lies inside the test follow-up
    and (2) does not exceed the train follow-up, as required by scikit-survival.


    Uses inner quantiles for numerical stability (avoids extremes where # at risk is tiny).
    """
    train_times = np.asarray(train_times, dtype=float)
    test_times = np.asarray(test_times, dtype=float)


    # Bounds inside both train and test support (see scikit-survival docs)
    lo_test = np.quantile(test_times, lower_q)
    hi_test = np.quantile(test_times, upper_q)
    hi_train = np.max(train_times)


    eps = 1e-8
    t_lo = max(lo_test, np.min(test_times[test_times > 0]) + eps)
    t_hi = min(hi_test, hi_train - eps, np.max(test_times) - eps)


    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
    # Fallback: use a conservative slice strictly inside the overlapping range
        t_lo = np.percentile(test_times, 20)
        t_hi = min(np.percentile(test_times, 80), hi_train - eps)
    if t_hi <= t_lo:
    # As a last resort, return 2 points to avoid errors downstream
        t_lo = max(np.min(test_times) + eps, 1.0)
        t_hi = max(t_lo + 1.0, np.max(test_times) - eps)

    return np.linspace(t_lo, t_hi, n_times)

# aggregaring folds for cumulative auc plotting
def _aggregate_cd_auc(per_fold_cd: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
                      n_times_common: int = 75) -> Dict[str, dict]:
    """Interpolate per-fold cumulative_dynamic_auc onto a common grid and
    return mean/std envelopes per model.

    per_fold_cd[model] = list of (times_k, auc_k) for each outer fold.
    """
    out: Dict[str, dict] = {}

    for model, fold_list in per_fold_cd.items():
        if not fold_list:
            continue

        # Validate and normalize each (times, auc) pair
        grids: List[Tuple[np.ndarray, np.ndarray]] = []
        for times_k, auc_k in fold_list:
            t = np.asarray(times_k, dtype=float).ravel()
            a = np.asarray(auc_k, dtype=float).ravel()
            if t.ndim != 1 or a.ndim != 1 or len(t) == 0 or len(t) != len(a):
                # Skip malformed entries
                continue
            grids.append((t, a))

        if not grids:
            continue

        # Build a common time grid that lies inside *all* fold grids
        lows = [t[0] for t, _ in grids]
        highs = [t[-1] for t, _ in grids]
        t_lo = max(lows)
        t_hi = min(highs)

        if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
            # Fallback to median overlap
            t_lo = float(np.median(lows))
            t_hi = float(np.median(highs))
        if t_hi <= t_lo:
            # Give up on this model if no overlap exists
            continue

        common_times = np.linspace(t_lo, t_hi, n_times_common)

        # Accumulate interpolated rows in a *list*, then stack once
        auc_rows: List[np.ndarray] = []
        for t, a in grids:
            auc_rows.append(np.interp(common_times, t, a))

        if not auc_rows:
            continue
        auc_mat = np.vstack(auc_rows)

        out[model] = {
            "times_days": common_times,
            "times_years": common_times / 365.0,
            "mean": auc_mat.mean(axis=0),
            "std": auc_mat.std(axis=0),
        }

    return out

# Preparing safe time grids for cumulative auc
def make_time_grid_quantile(y_train_struct, y_test_struct, max_points: int = 40,
                            lo_q: float = 0.10, hi_q: float = 0.90) -> np.ndarray:
    ev_times = np.asarray(y_test_struct["time"][y_test_struct["event"]], dtype=float)
    if ev_times.size == 0:
        return np.array([])

    # Trim extremes on the test events and cap by train max time (required by sksurv)
    lo = np.quantile(ev_times, lo_q)
    hi = np.quantile(ev_times, hi_q)
    hi = min(hi, float(np.max(y_train_struct["time"])) - 1e-8)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([])

    # Use quantiles to place points where data exist
    uniq = np.unique(ev_times[(ev_times >= lo) & (ev_times <= hi)])
    if uniq.size == 0:
        return np.array([])
    k = int(min(max_points, max(10, uniq.size)))  # 10–40 points typical
    qs = np.linspace(lo_q, hi_q, k)
    grid = np.quantile(ev_times, qs)
    return np.unique(grid)

# Filtering time grids for cumulative auc to avoid nans
def filter_auc_grid_by_comparable_pairs(y_test_struct, time_grid: np.ndarray) -> np.ndarray:
    if time_grid.size == 0:
        return time_grid
    times = np.asarray(y_test_struct["time"], dtype=float)
    events = np.asarray(y_test_struct["event"], dtype=bool)

    t_col = time_grid[:, None]    # (T,1)
    times_row = times[None, :]    # (1,N)
    events_row = events[None, :]  # (1,N)

    cases = ((events_row) & (times_row <= t_col)).sum(axis=1)  # events by t
    at_risk = (times_row >= t_col).sum(axis=1)                 # still at risk at t
    controls = at_risk - cases

    mask = (cases > 0) & (controls > 0)
    return time_grid[mask]

# Safely using sksurv
def compute_cd_auc_robust(y_train_struct, y_test_struct, risk_scores: np.ndarray,
                          base_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times, auc_t) with invalid times removed. If <2 valid points, return ([], [])."""
    t_eval = filter_auc_grid_by_comparable_pairs(y_test_struct, base_grid)
    if t_eval.size < 2:
        return np.array([]), np.array([])

    auc_t, _ = cumulative_dynamic_auc(y_train_struct, y_test_struct, risk_scores, t_eval)
    auc_t = np.asarray(auc_t, dtype=float)

    # Drop any remaining non-finite values and align times
    m = np.isfinite(auc_t)
    t_eval, auc_t = t_eval[m], auc_t[m]
    if t_eval.size < 2:
        return np.array([]), np.array([])
    return t_eval, auc_t

###################################
### Defining Plotting Functions ###
###################################

# Function to plot mean ROC curves
def plot_mean_roc(curves_summary, metrics_dict, savepath=None):
    plt.figure(figsize=(7,6))
    for model_name, (fpr, mean_tpr) in curves_summary['roc'].items():
        mean_auc = np.mean([m['roc_auc'] for m in metrics_dict[model_name]])
        plt.plot(fpr, mean_tpr, label=f'{model_name} (AUC={mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title("CV Pooled ROC-Curves", fontsize=18)
    plt.tick_params(axis="both", labelsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

# Function to plot mean PR curves
def plot_mean_pr(curves_summary, metrics_dict, savepath=None):
    plt.figure(figsize=(7,6))
    for model_name, (recall, mean_prec) in curves_summary['pr'].items():
        mean_ap = np.mean([m['pr_auc'] for m in metrics_dict[model_name]])
        plt.plot(recall, mean_prec, label=f'{model_name} (AP={mean_ap:.2f})')
    plt.hlines(y.sum()/len(y), 0, 1, colors="k", linestyles="--", label="Baseline")
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title("CV Pooled PR-Curves", fontsize=18)
    plt.tick_params(axis="both", labelsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def plot_km_curves(survival_results, max_years=5, savepath=None):
    risk_labels = {0: "Low Risk", 1: "High Risk"}
    model_list = list(survival_results.keys())
    colours = ["#C190F0", "#35AB6A"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    logrank_ps = []
    iteration = 0
    for idx, model_name in enumerate(model_list):
        ax = axes[idx]

        # Pool all folds
        dfs = []
        for dss_val, pred_class in survival_results[model_name]:
            temp = dss_val.copy()
            temp["pred_class"] = pred_class
            dfs.append(temp)
        pooled_df = pd.concat(dfs, ignore_index=True)

        high = pooled_df[pooled_df["pred_class"] == 1]
        low  = pooled_df[pooled_df["pred_class"] == 0]
        lr_res = logrank_test(
            high["DSS.time"], low["DSS.time"],
            event_observed_A=high["DSS"], event_observed_B=low["DSS"]
        )
        p_val = lr_res.p_value
        logrank_ps.append(p_val)
        fdr_vals = multipletests(logrank_ps, method="fdr_bh")[1]
        ax.set_title(f"{model_name}\nLog-Rank p FDR = {fdr_vals[iteration]:.2e}", fontsize=16)
        iteration += 1
        # KM curves
        kmf = KaplanMeierFitter()
        for cls in [0, 1]:
            mask = pooled_df["pred_class"] == cls
            n_value = mask.sum()
            kmf.fit(
                durations=pooled_df.loc[mask, "DSS.time"] / 365,
                event_observed=pooled_df.loc[mask, "DSS"],
                label=f"{risk_labels[cls]} (n={n_value})"
            )
            kmf.plot_survival_function(ax=ax, ci_show=True, color=colours[cls])

        ax.set_xlim(0, max_years)
        ax.set_xlabel("Time (years)", fontsize=16)
        ax.set_ylabel("Survival probability", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=10,loc="lower left",edgecolor="black")

    # Remove unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def plot_forest(survival_results, metrics_summary_dict, savepath=None):
    """
    Plot Concordance (left) | HR (forest) (middle) | metrics table (right).
    Model names appear on the leftmost axis (concordance). Continuous separators
    span the full figure so lines meet across panels.
    """
    from matplotlib.lines import Line2D
    results = []
    model_list_local = list(survival_results.keys())

    # Fit Cox model per model and collect HR / CI / p and c-index (assumes cph.concordance_index_ set)
    for model_name in model_list_local:
        dfs = []
        for dss_val, pred_class in survival_results[model_name]:
            temp = dss_val.copy()
            temp["pred_class"] = pred_class
            dfs.append(temp)
        pooled_df = pd.concat(dfs, ignore_index=True)

        cph_df = pooled_df.rename(columns={"DSS.time": "time", "DSS": "event"})
        cph = CoxPHFitter()
        cph.fit(cph_df, duration_col="time", event_col="event")
        summary = cph.summary.loc["pred_class"]

        c_index = getattr(cph, "concordance_index_", None)

        results.append({
            'Model': model_name,
            'HR': summary['exp(coef)'],
            'CI_lower': summary['exp(coef) lower 95%'],
            'CI_upper': summary['exp(coef) upper 95%'],
            'p': summary['p'],
            'cindex': c_index
        })

    sig_df = pd.DataFrame(results)

    # Build metrics table
    rows = []
    for model_name in model_list_local:
        row = {'Model': model_name}
        ms = metrics_summary_dict.get(model_name, {})
        if ms:
            def to_tuple(x):
                return x if isinstance(x, (list, tuple, np.ndarray)) else (x, np.nan)
            row.update({
                'accuracy_mean': to_tuple(ms.get('accuracy', (np.nan, np.nan)))[0],
                'accuracy_std' : to_tuple(ms.get('accuracy', (np.nan, np.nan)))[1],
                'precision_mean': to_tuple(ms.get('precision', (np.nan, np.nan)))[0],
                'precision_std' : to_tuple(ms.get('precision', (np.nan, np.nan)))[1],
                'recall_mean': to_tuple(ms.get('recall', (np.nan, np.nan)))[0],
                'recall_std' : to_tuple(ms.get('recall', (np.nan, np.nan)))[1],
                'f1_mean': to_tuple(ms.get('f1', (np.nan, np.nan)))[0],
                'f1_std' : to_tuple(ms.get('f1', (np.nan, np.nan)))[1],
                'roc_auc_mean': to_tuple(ms.get('roc_auc', (np.nan, np.nan)))[0],
                'roc_auc_std' : to_tuple(ms.get('roc_auc', (np.nan, np.nan)))[1],
                'pr_auc_mean': to_tuple(ms.get('pr_auc', (np.nan, np.nan)))[0],
                'pr_auc_std' : to_tuple(ms.get('pr_auc', (np.nan, np.nan)))[1],
            })
        rows.append(row)
    metrics_df = pd.DataFrame(rows)

    # Merge and compute FDR / significance
    sig_df = pd.merge(sig_df, metrics_df, on="Model", how="left")
    sig_df["FDR"] = multipletests(sig_df["p"], method="fdr_bh")[1]
    sig_df["Sig"] = sig_df["FDR"] < 0.05
    savepath_csv = savepath+".csv"
    sig_df.to_csv(savepath_csv)
    # Keep original order
    ordered_models = [m for m in model_list_local if m in sig_df["Model"].values]
    df_forest = sig_df.set_index("Model").loc[ordered_models].reset_index()

    # y positions
    n_models = len(df_forest)
    y = np.arange(n_models)

    # Figure & GridSpec (left: concordance, mid: HR, right: table)
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 3.0], wspace=0.075)

    # Create axes in the new order: concordance left, HR middle, table right
    ax_cindex = fig.add_subplot(gs[0])
    ax_hr = fig.add_subplot(gs[1])
    ax_table = fig.add_subplot(gs[2])
    ax_table.set_axis_off()

    # ---------- Concordance (left) ----------
    ax_cindex.barh(y, df_forest["cindex"], align='center', height=0.6, zorder=2,
                   edgecolor="black", color="#7FC97F")
    ax_cindex.set_xlim(0, 1)
    ax_cindex.set_xlabel("C-index", fontsize=12)
    ax_cindex.set_xticks(ticks=[0.0,0.2,0.4,0.6,0.8,1.0])
    # put model names on the LEFT axis (concordance)
    ax_cindex.set_yticks(y)
    ax_cindex.set_yticklabels(df_forest["Model"].tolist(), fontsize=11)
    ax_cindex.yaxis.set_ticks_position('left')
    ax_cindex.yaxis.set_label_position('left')
    ax_cindex.tick_params(axis='y', labelleft=True)
    ax_cindex.invert_yaxis()
    ax_cindex.set_title("Concordance", fontsize=12, fontweight='bold')
    ax_cindex.tick_params(axis='x', labelsize=10)

    for yi, cval in zip(y, df_forest["cindex"]):
        # annotate c-index values to the right of bars
        if not np.isnan(cval):
            ax_cindex.text(cval + 0.01, yi, f"{cval:.2f}", va='center', fontsize=9)

    # ---------- Hazard Ratio (forest) (middle) ----------
    hr_err_low = df_forest["HR"] - df_forest["CI_lower"]
    hr_err_high = df_forest["CI_upper"] - df_forest["HR"]
    ax_hr.errorbar(df_forest["HR"], y,
                   xerr=[hr_err_low, hr_err_high],
                   fmt='none', ecolor='black', capsize=5, elinewidth=1.5, capthick=1.5, zorder=1)
    ax_hr.scatter(df_forest["HR"], y,
                  c=df_forest["Sig"], cmap='Set2_r',
                  linewidths=1.5, edgecolors='black', zorder=2, vmin=0, s=150)

    # hide y labels on the middle HR axis (they are on the left axis)
    ax_hr.set_yticks(y)
    ax_hr.set_yticklabels([])
    ax_hr.tick_params(axis='y', length=0)

    # set consistent y-limits on all axes then invert for plotting order
    ymin, ymax = -0.5, n_models - 0.5
    ax_cindex.set_ylim(ymin, ymax)
    ax_hr.set_ylim(ymin, ymax)
    ax_table.set_ylim(ymin, ymax)
    ax_cindex.invert_yaxis()
    ax_hr.invert_yaxis()
    ax_table.invert_yaxis()

    ax_hr.axvline(1, color='red', linestyle='--')
    hr_max = np.nanmax(df_forest[["CI_upper", "HR"]].values) * 1.1
    hr_xlim = max(8, hr_max)
    ax_hr.set_xlim(0, hr_xlim)
    ax_hr.set_xlabel("Hazard Ratio (HR)", fontsize=12)
    ax_hr.set_title("Forest Plot", fontsize=14, fontweight='bold')
    ax_hr.tick_params(axis='x', labelsize=10)

    # ---------- Metrics table (text) (right) ----------
    headers = ["HR", "FDR", "ACC", "PRC", "REC", "F1", "ROC", "AP"]
    col_x = np.linspace(0, 1, len(headers))

    # header: just above the top row (top is at y = 0 after inversion)
    header_y = ymin - 0.05
    for x, h in zip(col_x, headers):
        ax_table.text(x, header_y, h, fontsize=12, fontweight="bold", ha="center", va="bottom")

    # metric rows aligned with y positions
    for row_idx, row in enumerate(df_forest.itertuples(), start=0):
        values = [
            f"{row.HR:.2f}",
            f"{row.FDR:.2e}",
            f"{row.accuracy_mean:.2f}\u00B1{row.accuracy_std:.2f}",
            f"{row.precision_mean:.2f}\u00B1{row.precision_std:.2f}",
            f"{row.recall_mean:.2f}\u00B1{row.recall_std:.2f}",
            f"{row.f1_mean:.2f}\u00B1{row.f1_std:.2f}",
            f"{row.roc_auc_mean:.2f}\u00B1{row.roc_auc_std:.2f}",
            f"{row.pr_auc_mean:.2f}\u00B1{row.pr_auc_std:.2f}"
        ]
        for x, val in zip(col_x, values):
            ax_table.text(x, row_idx, val, fontsize=10, ha="center", va="center")

    # horizontal separators for readability
    for yi in range(0,5):
        ax_hr.hlines(yi + 0.5, xmin=ax_hr.get_xlim()[0], xmax=ax_hr.get_xlim()[1],
                     colors='lightgray', linestyles='--', linewidth=1, zorder=0)
        ax_cindex.hlines(yi + 0.5, xmin=ax_cindex.get_xlim()[0], xmax=ax_cindex.get_xlim()[1],
                         colors='lightgray', linestyles='--', linewidth=1, zorder=0)

    # ---------- Continuous separators across all panels ----------
    # Draw full-figure separators so they meet across axes
    fig.canvas.draw()
    x_left_fig = ax_cindex.get_position().x0
    x_right_fig = ax_table.get_position().x1
    for yi in range(0,5):
        y_data = yi + 0.5
        _, y_disp = ax_cindex.transData.transform((0, y_data))
        _, y_fig = fig.transFigure.inverted().transform((0, y_disp))
        line = Line2D([x_left_fig, x_right_fig+0.025], [y_fig, y_fig], transform=fig.transFigure,
                      color='lightgray', linestyle='--', linewidth=1, zorder=0)
        fig.add_artist(line)

    # finalize
    savepath_plot = savepath+".png"
    plt.savefig(savepath_plot)
    plt.close()

# Plotting Cumulative Dynamic AUC
def plot_mean_cumulative_dynamic_auc(cauc_agg: Dict[str, dict],
    savepath: str,
    time_unit: str = "years") -> None:
    """Plot mean ± std cumulative_dynamic_auc across folds for each model.
    time_unit: "years" (default) or "days".
    """
    plt.figure(figsize=(7, 6))
    for model, stats in cauc_agg.items():
        if time_unit == "years":
            tx = stats["times_years"]
            xlab = "Time (years)"
        else:
            tx = stats["times_days"]
            xlab = "Time (days)"
        mu = stats["mean"]
        sd = stats["std"]
        plt.plot(tx, mu, lw=2, label=model,marker="o", markersize=5,markevery=1)
        plt.fill_between(tx, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), alpha=0.20)

    plt.ylim(0.0, 1.0)
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel("Time-dependent AUC (mean ± sd across folds)", fontsize=16)
    plt.tick_params(axis="both", labelsize=14)
    plt.title("Cumulative/Dynamic AUC", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

######################################
### Defining the Training Function ###
######################################
def train_evaluate_model(random_state=42,outer_folds=3,inner_folds=3,inner_iterations=25,ANN_iterations=25,save_dir="./LGG_Prognosis_Results",dataset_id=0):

    RANDOM_STATE = random_state
    OUTER_FOLDS = outer_folds
    INNER_FOLDS = inner_folds
    N_ITER_INNER = inner_iterations
    N_ITER_ANN = ANN_iterations
    N_JOBS = -1

    DATA_URL = (f"https://raw.githubusercontent.com/JackWJW/LGG_Prognosis_Prediction/main/Tidied_Datasets/tidied_integrated_df_{dataset_id}.csv")
    data = pd.read_csv(DATA_URL).drop(columns=["Unnamed: 0"])
    dss_info = data[["DSS", "DSS.time"]]

    X = data.drop(columns = ["Srv", "DSS", "DSS.time"])
    y = LabelEncoder().fit_transform(data["Srv"])
    with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
        print('Starting nested cross-validation...',file=file)

    # Defining Folds
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Preparing storage dictionaries
    models_info = {name: {'estimator': m, 'space': s} for name, (m, s) in search_spaces.items()}
    models_info['ANN'] = {'estimator': net, 'space': deep_search_space}

    # Out of fold storage
    oof_results = {
        name: {
            'y_true': np.zeros(len(y), dtype=int),
            'probs': np.full(len(y), np.nan, dtype=float),
            'Predicted Class': np.full(len(y), -1, dtype=int)
        } for name in models_info
    }
    oof_results['Ensemble (Mean)'] = {'y_true': np.zeros(len(y), dtype=int),
                                    'probs': np.full(len(y), np.nan, dtype=float),
                                    'Predicted Class': np.full(len(y), -1, dtype=int)}

    # Storing per fold metrics and curves
    per_fold_metrics = {name: [] for name in models_info}
    per_fold_curves = {name: {'roc': [], 'pr': []} for name in models_info}

    # Storing the per fold probabilities
    per_fold_probs = {name: [] for name in models_info}
    per_fold_probs['Ensemble (Mean)'] = []

    # Setting up Ensemble placeholders
    per_fold_metrics['Ensemble (Mean)'] = []
    per_fold_curves['Ensemble (Mean)'] = {'roc': [], 'pr': []}

    #Setting up stoage dictionaries for threshold tuning
    per_fold_train_probs = {name: [] for name in models_info}
    per_fold_tuned_thresholds = {name: [] for name in models_info}
    per_fold_tuned_thresholds['Ensemble (Mean)'] = []
    per_fold_train_probs['Ensemble (Mean)'] = []

    #Storage for cumulative auc curves
    per_fold_cd_auc = {name: [] for name in models_info}
    per_fold_cd_auc['Ensemble (Mean)'] = []

    # Iterate over the outer folds
    model_list = list(models_info.keys())
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
            print(f'\nOuter Fold: {fold_idx+1}/{OUTER_FOLDS}',file=file)
        SAVE_DIR = f"./LGG_Prognosis_Results/RS-{rs_number}_DS-{dataset_id}_Results/ANN_Training_{fold_idx+1}"
        X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_train, y_test = y[train_idx], y[test_idx]

        train_event = dss_info['DSS'].values[train_idx].astype(bool)
        train_time  = dss_info['DSS.time'].values[train_idx].astype(float)
        test_event  = dss_info['DSS'].values[test_idx].astype(bool)
        test_time   = dss_info['DSS.time'].values[test_idx].astype(float)
        y_train_struct = Surv.from_arrays(event=train_event, time=train_time)
        y_test_struct  = Surv.from_arrays(event=test_event, time=test_time)
        fold_time_grid = make_time_grid_quantile(y_train_struct, y_test_struct, max_points=40)

        for model_name, info in models_info.items():
            with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                print(f'\nTuning and Fitting: {model_name}',file=file)
            base = clone(info['estimator'])
            space = info['space']

            # Defining the pipeline for the model
            if model_name == 'ANN':
                pipe = Pipeline([
                    ('low_var', VarianceThreshold(threshold=0.01)),
                    ('kbest', SelectKBest(score_func=f_classif, k=50)),
                    ('scaler', StandardScaler()),
                    ('smote', SMOTETomek(random_state=RANDOM_STATE)),
                    ('clf', base)
                ])

            else:
                pipe = Pipeline([
                    ('low_var', VarianceThreshold(threshold=0.01)),
                    ('kbest', SelectKBest(score_func=f_classif, k=50)),
                    ('scaler', StandardScaler()),
                    ('clf', base)
                ])

            # prefix search space
            space_prefixed = {f'clf__{k}': v for k, v in space.items()}
            space_prefixed['kbest__k'] = Integer(20, 65)

            #Selecting iterations
            n_iter = N_ITER_ANN if model_name == 'ANN' else N_ITER_INNER
            n_jobs = 1 if model_name == 'ANN' else N_JOBS

            opt=BayesSearchCV(
                estimator=pipe,
                search_spaces=space_prefixed,
                n_iter=n_iter,
                scoring='average_precision',
                cv=INNER_FOLDS,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
                refit=True
            )

            # Fitting
            fit_X = X_train.astype(np.float32) if model_name == 'ANN' else X_train
            try:
                opt.fit(fit_X, y_train)
                # Plotting Loss Curves
                if model_name == "ANN":
                    best = opt.best_estimator_
                    
                    # skorch Net is inside the pipeline as step 'clf'
                    net_trained = best.named_steps['clf']

                    # Access the training history
                    history = net_trained.history_

                    # Extract values
                    train_losses = history[:, 'train_loss']
                    valid_losses = history[:, 'valid_loss']
                    plt.figure(figsize=(6,4))
                    plt.plot(train_losses, label="Train Loss")
                    if valid_losses is not None:
                        plt.plot(valid_losses, label="Valid Loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Fold {fold_idx+1} - ANN Loss Curves")
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.savefig(SAVE_DIR)
                    plt.close()

                with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:    
                    print(f"    Best params for {model_name} (fold {fold_idx+1}): {opt.best_params_}",file=file)
            except Exception as e:
                with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                    print(f'BayesSearchCV failed for {model_name} in fold {fold_idx+1}: {e}',file=file)
                # In this case fit base estimator inside the pipeline without search
                pipe.set_params(**{})
                pipe.fit(fit_X, y_train)
                opt=None

            # Get the best estimator
            if opt is not None:
                best = opt.best_estimator_
            else:
                best = pipe
            
            # Tuning for best threshold:
            train_preds_input = fit_X.astype(np.float32) if model_name == 'ANN' else fit_X
            probs_train = best.predict_proba(train_preds_input)[:, 1].ravel()
            per_fold_train_probs[model_name].append({'train_idx': train_idx, 'probs': probs_train, 'y_true': y_train})

            thr, thr_ba = tune_threshold(probs_train, y_train)
            per_fold_tuned_thresholds[model_name].append(thr)
            with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                print(f"    Tuned threshold for {model_name} (fold {fold_idx+1}): {thr:.2f} (Balanced Accuracy={thr_ba:.3f})",file=file)

            # Predict proba on the test set
            test_X = X_test.astype(np.float32) if model_name == 'ANN' else X_test
            probs = best.predict_proba(test_X)[:, 1].ravel()

            # Preparation for cumulative dynamic plotting
            try:
                t_eval, auc_vec = compute_cd_auc_robust(y_train_struct, y_test_struct, probs, fold_time_grid)
                if t_eval.size >= 2:
                    per_fold_cd_auc[model_name].append((t_eval, auc_vec))
                    with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                        print(f"    cumulative_dynamic_auc stored (fold {fold_idx+1}): {auc_vec.mean():.3f} over {t_eval.size} t's", file=file)
                else:
                    with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                        print(f"    cumulative_dynamic_auc skipped (fold {fold_idx+1}): insufficient comparable times", file=file)
            except Exception as e:
                with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                    print(f"    cumulative_dynamic_auc failed for {model_name} on fold {fold_idx+1}: {e}",file=file)

            # Storing out of fold probabilities
            oof_results[model_name]['probs'][test_idx] = probs
            oof_results[model_name]['y_true'][test_idx] = y_test
            
            # Storing out of fold predictions
            model_predictions = (probs >= thr).astype(int)
            oof_results[model_name]['Predicted Class'][test_idx] = model_predictions

            # Store per fold probabilities for ensemble
            per_fold_probs[model_name].append({'test_idx': test_idx, 'probs': probs, 'y_true': y_test})

            # Compute per-fold metrics and curves
            m = compute_metrics(y_test, probs, threshold=thr)
            per_fold_metrics[model_name].append(m)
            fpr, tpr, _ = roc_curve(y_test, probs)
            prec, rec, _ = precision_recall_curve(y_test, probs)
            per_fold_curves[model_name]['roc'].append((fpr, tpr))
            per_fold_curves[model_name]['pr'].append((rec, prec))

            with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                print(f"    Fold {fold_idx+1} {model_name}: AP={m['pr_auc']:.3f}, ROC AUC={m['roc_auc']:.3f}", file=file)
        
        # Computing Ensemble predictions for this fold against the train set for threshold tuning
        probs_stack_train = []
        for m in model_list:
            entry = per_fold_train_probs[m][-1]   # matches current fold
            probs_stack_train.append(entry['probs'])
        probs_stack_train = np.vstack(probs_stack_train)

        ensemble_train_mean = np.mean(probs_stack_train, axis=0)

        # Tuning Ensemble threshold on ensemble_train_mean
        thr_ens, thr_ens_ba = tune_threshold(ensemble_train_mean, y_train)
        per_fold_tuned_thresholds['Ensemble (Mean)'].append(thr_ens)
        per_fold_train_probs['Ensemble (Mean)'].append({'train_idx': train_idx, 'probs': ensemble_train_mean, 'y_true': y_train})

        with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
            print("\nTuning Ensemble:",file=file)
            print(f"    Tuned threshold for Ensemble (fold {fold_idx+1}): {thr_ens:.2f} (Balanced Accuracy={thr_ens_ba:.3f})",file=file)
        # Computing Ensemble predictions for this fold against the test set
        probs_stack_test = []
        for m in model_list:
            probs_stack_test.append(per_fold_probs[m][-1]['probs'])
        probs_stack_test = np.vstack(probs_stack_test)
        ensemble_mean_fold = np.mean(probs_stack_test, axis=0)

        try:
            t_eval, auc_vec = compute_cd_auc_robust(y_train_struct, y_test_struct, probs, fold_time_grid)
            if t_eval.size >= 2:
                per_fold_cd_auc['Ensemble (Mean)'].append((t_eval, auc_vec))
                with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                    print(f"    cumulative_dynamic_auc stored (fold {fold_idx+1}): {auc_vec.mean():.3f} over {t_eval.size} t's", file=file)
            else:
                with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                    print(f"    cumulative_dynamic_auc skipped (fold {fold_idx+1}): insufficient comparable times", file=file)
        except Exception as e:
            with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
                print(f"    cumulative_dynamic_auc failed for Ensemble on fold {fold_idx+1}: {e}",file=file)

        y_true_fold = per_fold_probs[model_list[0]][-1]['y_true']

        per_fold_probs['Ensemble (Mean)'].append({'test_idx': test_idx, 'probs': ensemble_mean_fold, 'y_true': y_true_fold})

        #Calculating ensemble predictions
        ensemble_preds = (ensemble_mean_fold >= thr_ens).astype(int)
        oof_results['Ensemble (Mean)']['Predicted Class'][test_idx] = ensemble_preds

        # Store ensemble OOF probs (if you keep same oof_results structure)
        oof_results['Ensemble (Mean)']['probs'][test_idx] = ensemble_mean_fold
        oof_results['Ensemble (Mean)']['y_true'][test_idx] = y_true_fold

        # Store ensemble metrics and curves
        m_mean = compute_metrics(y_true_fold, ensemble_mean_fold, threshold=thr_ens)
        per_fold_metrics['Ensemble (Mean)'].append(m_mean)
        fpr, tpr, _ = roc_curve(y_true_fold, ensemble_mean_fold)
        prec, rec, _ = precision_recall_curve(y_true_fold, ensemble_mean_fold)
        per_fold_curves['Ensemble (Mean)']['roc'].append((fpr, tpr))
        per_fold_curves['Ensemble (Mean)']['pr'].append((rec, prec))
    
    base_models = [m for m in oof_results.keys()]
    oof_df = pd.DataFrame({'y_true': oof_results[base_models[0]]['y_true']})

    oof_df['DSS.time'] = dss_info['DSS.time']
    oof_df['DSS'] = dss_info['DSS']

    for m in base_models:
        oof_df[f'prob_{m}'] = oof_results[m]['probs']
        oof_df[f'pred_{m}'] = oof_results[m]['Predicted Class']
    
    metrics_summary = {}
    for m in model_list + ['Ensemble (Mean)']:
        vals = per_fold_metrics[m]
        summary = {k: (np.mean([v[k] for v in vals]), np.std([v[k] for v in vals])) for k in vals[0].keys()}
        metrics_summary[m] = summary
    curves_summary = {'roc': {}, 'pr': {}}
    all_models_for_plot = model_list + ['Ensemble (Mean)']
    for m in all_models_for_plot:
        if m.startswith("Ensemble"):
            y_probs = oof_df['prob_Ensemble (Mean)'].values
        else:
            y_probs = oof_df[f'prob_{m}'].values
        y_true = oof_df['y_true'].values

        curves_summary['roc'][m] = compute_pooled_curve([y_probs], [y_true], curve_type="roc")
        curves_summary['pr'][m]  = compute_pooled_curve([y_probs], [y_true], curve_type="pr")

    # Prepare metrics dictionary
    metrics_for_plot = {m: per_fold_metrics.get(m, []) for m in model_list}
    metrics_for_plot['Ensemble (Mean)'] = per_fold_metrics.get('Ensemble (Mean)', [])

    oof_df['DSS.time'] = dss_info['DSS.time']
    oof_df['DSS'] = dss_info['DSS']

    # Preparing survival resutls dictionaries
    survival_results = {}
    for m in base_models:
        survival_results[m] = [(oof_df[['DSS.time', 'DSS']], oof_df[f'pred_{m}'])]
    
    # Preparing results for cumulative auc
    cauc_agg = _aggregate_cd_auc(per_fold_cd_auc, n_times_common=50)

    return oof_df, metrics_summary, curves_summary, metrics_for_plot, survival_results, y, cauc_agg

##########################################
### Model Training and Evaluation Loop ###
##########################################

for rs_number in range(0,3):
    for dataset_id in range(1,19):
        with open("./LGG_Prognosis_Results/training_log.txt", "a") as file:
            print(f"\nStarting training run for Random State = {rs_number} and Dataset ID = {dataset_id}\n", file=file)
        directory = f"./LGG_Prognosis_Results/RS-{rs_number}_DS-{dataset_id}_Results"
        os.makedirs(directory)
        oof_df, metrics_summary, curves_summary, metrics_for_plot, survival_results, y, cauc_agg= train_evaluate_model(random_state=rs_number, outer_folds=3,inner_folds=3,inner_iterations=25,ANN_iterations=25, dataset_id=dataset_id, save_dir=f"./LGG_Prognosis_Results/RS-{rs_number}_DS-{dataset_id}_Results")

        plot_mean_roc(curves_summary, metrics_for_plot,savepath=f"./LGG_Prognosis_Results/RS-{rs_number}_DS-{dataset_id}_Results/ROC-AUC.png")
        plot_mean_pr(curves_summary, metrics_for_plot,savepath=f"./LGG_Prognosis_Results/RS-{rs_number}_DS-{dataset_id}_Results/PR-AUC.png")

        plot_km_curves(survival_results, savepath=f"./LGG_Prognosis_Results/RS-{rs_number}_DS-{dataset_id}_Results/KM_plots.png")
        plot_forest(survival_results,metrics_summary,savepath=f"./LGG_Prognosis_Results/RS-{rs_number}_DS-{dataset_id}_Results/Results_Summary")
        plot_mean_cumulative_dynamic_auc(cauc_agg, savepath=f"./LGG_Prognosis_Results/RS-{rs_number}_DS-{dataset_id}_Results/Cumulative_Dynamic_AUC.png",time_unit="years",)

        oof_df.to_csv(f"./LGG_Prognosis_Results/RS-{rs_number}_DS-{dataset_id}_Results/Predictions_Probabilities.csv")