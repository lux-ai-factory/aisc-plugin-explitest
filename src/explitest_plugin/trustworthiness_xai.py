"""

Bridge tabular data/explanations to FA multivariate time-series metrics by
adding a singleton time axis: (B, F) -> (B, 1, F)

Saves CSV results for each (model, explainer, metric):
- Per-sample scores table
- Summary table (mean/std/min/max + NaN/Inf counts)
"""

from typing import Any, Tuple, List, Dict, Optional, Callable, Protocol
import numpy as np
import pandas as pd

from ExpliTest import get_metric, list_metrics
from ExpliTest.core.enums import DataType, ExplanationType

from explainers import ExplainerFactory


class BaseEstimatorProtocol(Protocol):
    def predict_proba(
        self, X: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]: ...


# ============================================================
# Expected input dataframe columns
# ============================================================
# df must contain:
#   Dataset (optional), Model, Explainer, Metric, Value
#
# - Value must be direction-aligned already:
#     higher = better for ALL metrics (including performance)
#   e.g. for RMSE you should pass -RMSE as Value
#
# - We infer Metric -> Category internally.
# - Performance is included as Metric(s) and treated like other categories:
#     normalize per metric, then mean within Performance category.

EXPLAINER_NAMES = [
    "tabular_ablation",
    "occlusion",
    "feature_ablation",
    "shapley_sampling",
    "lime_tabular",
    "kernel_shap",
    "sampling_shap",
    "shap",
]

FAITHFULNESS_METRICS = {
    "fa_ts_mv_faithfulness_correlation",
    "fa_ts_mv_pixel_flipping",
}

ROBUSTNESS_METRICS = {
    "fa_ts_mv_avg_sensitivity",
    "fa_ts_mv_continuity",
}

COMPLEXITY_METRICS = {
    "fa_ts_mv_sparseness_element",
    "fa_ts_mv_sparseness_channel",
    "fa_ts_mv_complexity_entropy_channel",
    "fa_ts_mv_complexity_entropy_element",
}

PERFORMANCE_METRICS = {
    "Accuracy",
    "RMSE",
    "F1-Score",
    # "AUC",
    # "MAE",
}

METRIC_TO_CATEGORY: Dict[str, str] = {}
METRIC_TO_CATEGORY.update({m: "Faithfulness" for m in FAITHFULNESS_METRICS})
METRIC_TO_CATEGORY.update({m: "Robustness" for m in ROBUSTNESS_METRICS})
METRIC_TO_CATEGORY.update({m: "Complexity" for m in COMPLEXITY_METRICS})
METRIC_TO_CATEGORY.update({m: "Performance" for m in PERFORMANCE_METRICS})

_METRIC_TO_CATEGORY_LOWER: Dict[str, str] = {
    k.lower(): v for k, v in METRIC_TO_CATEGORY.items()
}


def _infer_metric_category(metric: Any) -> Optional[str]:
    if metric is None:
        return None
    key = str(metric).strip()
    return METRIC_TO_CATEGORY.get(key) or _METRIC_TO_CATEGORY_LOWER.get(key.lower())


def _minmax_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if not s.notna().any():
        return pd.Series(np.nan, index=s.index)
    mn = np.nanmin(s.values)
    mx = np.nanmax(s.values)
    if not np.isfinite(mn) or not np.isfinite(mx):
        return pd.Series(np.nan, index=s.index)
    if mx == mn:
        # FIXME: for values between 0 and 1, returning 1 is suboptimal
        return s if 0 <= mn <= 1 else pd.Series(1.0, index=s.index)
    return (s - mn) / (mx - mn)


def compute_weighted_scores(
    df: pd.DataFrame,
    coeffs: Dict[str, float],
    *,
    dataset_col: Optional[str] = "dataset",
    model_col: str = "model",
    explainer_col: str = "explainer",
    metric_col: str = "metric",
    value_col: str = "mean",
    strict_metric_mapping: bool = True,
    normalize_per_dataset: bool = True,
) -> pd.DataFrame:
    """
    Produces one row per (Model, Explainer) (or per (Dataset,Model,Explainer) if Dataset exists)
    with:
      - Faithfulness / Robustness / Complexity / Performance:
          mean of per-metric min-max normalized values within that category
      - final_score:
          T*(F*Faith + R*Rob + C*Comp) + P*Perf

    Normalization:
      - Value is min-max normalized PER Metric.
      - If dataset_col exists and normalize_per_dataset=True:
            min-max is done per (Dataset, Metric)
        else:
            per Metric over the entire df.

    IMPORTANT:
      - "Performance" is treated like any other category:
        it can have multiple performance metrics (Accuracy, -RMSE, etc.)
        and we take the mean of their normalized values.
    """
    base_required = {model_col, explainer_col, metric_col, value_col}
    missing = base_required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    has_dataset = dataset_col is not None and dataset_col in df.columns

    group_keys = [model_col, explainer_col]
    if has_dataset:
        group_keys = [dataset_col] + group_keys

    work_cols = group_keys + [metric_col, value_col]
    work = df[work_cols].copy()

    work["_metric_category"] = work[metric_col].map(_infer_metric_category)

    unknown = work["_metric_category"].isna()
    if unknown.any():
        unknown_metrics = sorted(
            set(work.loc[unknown, metric_col].astype(str).tolist())
        )
        msg = f"Unknown metric(s) with no category mapping: {unknown_metrics}"
        if strict_metric_mapping:
            raise ValueError(msg)
        work = work.loc[~unknown].copy()

    if has_dataset and normalize_per_dataset:
        work["metric_norm"] = work.groupby([dataset_col, metric_col], dropna=False)[
            value_col
        ].transform(_minmax_series)
    else:
        work["metric_norm"] = work.groupby([metric_col], dropna=False)[
            value_col
        ].transform(_minmax_series)

    cat_means = (
        work.groupby(group_keys + ["_metric_category"], dropna=False)
        .agg(cat_mean=("metric_norm", "mean"))
        .reset_index()
    )

    out = cat_means.pivot_table(
        index=group_keys,
        columns="_metric_category",
        values="cat_mean",
        aggfunc="first",
    ).reset_index()

    for col in ["Faithfulness", "Robustness", "Complexity", "Performance"]:
        if col not in out.columns:
            out[col] = np.nan

    F = float(coeffs.get("F", 0.0))
    R = float(coeffs.get("R", 0.0))
    C = float(coeffs.get("C", 0.0))
    P = float(coeffs.get("P", 0.0))

    if "T" in coeffs:
        T = float(coeffs["T"])
    else:
        T = 1.0 if (F != 0.0 or R != 0.0 or C != 0.0) else 0.0

    faith = out["Faithfulness"].fillna(0.0)
    rob = out["Robustness"].fillna(0.0)
    comp = out["Complexity"].fillna(0.0)
    perf = out["Performance"].fillna(0.0)

    out["xai_component"] = (F * faith) + (R * rob) + (C * comp)
    out["xai_gated"] = T * out["xai_component"]
    out["perf_component"] = P * perf
    out["final_score"] = out["xai_gated"] + out["perf_component"]

    if (F == 0.0) and (R == 0.0) and (C == 0.0) and (P == 0.0):
        out["final_score"] = np.nan
        out["note"] = "No coefficients selected (F,R,C,P all zero)."
    else:
        out["note"] = ""

    sort_cols = ["final_score"] + group_keys
    out = out.sort_values(
        sort_cols,
        ascending=[False] + [True] * len(group_keys),
        na_position="last",
    ).reset_index(drop=True)
    return out


def rank_and_select_top_k(
    df: pd.DataFrame,
    coeffs: Dict[str, float],
    k: int,
    *,
    dataset_col: Optional[str] = "dataset",
    model_col: str = "model",
    explainer_col: str = "explainer",
    metric_col: str = "metric",
    value_col: str = "mean",
    strict_metric_mapping: bool = True,
    normalize_per_dataset: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]]]:
    """
    Returns:
      ranking_all  : one row per group with final_score and rank
      ranking_topk : top-k subset
      topk_pairs   : list[(Model, Explainer)] (dataset not included on purpose)
    """
    if k is None or int(k) < 0:
        raise ValueError("k must be a non-negative integer.")
    k = int(k)

    ranking_all = compute_weighted_scores(
        df,
        coeffs,
        dataset_col=dataset_col,
        model_col=model_col,
        explainer_col=explainer_col,
        metric_col=metric_col,
        value_col=value_col,
        strict_metric_mapping=strict_metric_mapping,
        normalize_per_dataset=normalize_per_dataset,
    ).copy()

    if ranking_all["final_score"].notna().any():
        ranking_all["rank"] = ranking_all["final_score"].rank(
            method="average", ascending=False
        )
    else:
        ranking_all["rank"] = np.nan

    eligible = ranking_all[ranking_all["final_score"].notna()].copy()
    ranking_topk = eligible.head(min(k, len(eligible))).reset_index(drop=True)

    topk_pairs = list(
        zip(
            ranking_topk[model_col].astype(str), ranking_topk[explainer_col].astype(str)
        )
    )
    return ranking_all, ranking_topk, topk_pairs


# Constructor params for metrics that need non-default values.
METRIC_PARAMS: dict[str, dict] = {
    "fa_ts_mv_faithfulness_correlation": {
        "kind": "classification",
        "subset_size": 4,
        "n_runs": 12,
        "seed": 42,
    },
    "fa_ts_mv_pixel_flipping": {"kind": "classification", "features_in_step": 1},
    "fa_ts_mv_avg_sensitivity": {
        "kind": "classification",
        "sigma": 0.01,
        "n_perturbations": 1,
        "seed": 42,
    },
    "fa_ts_mv_continuity": {
        "kind": "classification",
        "sigma": 0.01,
        "n_perturbations": 1,
        "seed": 42,
    },
}

# Robustness metrics that require explain_fn passed to evaluate().
NEEDS_EXPLAIN_FN = {"fa_ts_mv_avg_sensitivity", "fa_ts_mv_continuity"}


def to_single_step_btf(x_bf: np.ndarray) -> np.ndarray:
    """Convert tabular (B, F) to TS-like (B, 1, F)."""
    x = np.asarray(x_bf, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected tabular shape (B, F), got {x.shape}")
    return x[:, np.newaxis, :]


class TabularAsSingleStepTSModel:
    """Adapter so TS metrics can call a tabular predictor with (B, T, F) input."""

    def __init__(self, predictor: Any):
        self.predictor = predictor
        maybe_classes = getattr(predictor, "classes_", None)
        self.classes_ = (
            np.asarray(maybe_classes, dtype=int) if maybe_classes is not None else None
        )

    def __call__(self, x_btf: np.ndarray) -> np.ndarray:
        x = np.asarray(x_btf, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"Expected shape (B, T, F), got {x.shape}")
        if x.shape[1] != 1:
            raise ValueError(
                f"This adapter expects singleton time axis T=1, got T={x.shape[1]}"
            )
        x_tab = x[:, 0, :]
        probs = np.asarray(self.predictor.predict_proba(x_tab), dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError(
                f"predict_proba must return shape (B, K), got {probs.shape}"
            )
        n_classes = int(probs.shape[1])
        if self.classes_ is None or int(self.classes_.shape[0]) != n_classes:
            self.classes_ = np.arange(n_classes, dtype=int)
        return probs


def get_explainer_kwargs(
    explainer_name: str,
    *,
    background_tabular: np.ndarray,
    batch_size: int,
    fast: bool,
) -> dict[str, Any]:
    """
    Return kwargs for each explainer.
    fast=True uses cheaper settings for robustness explain_fn.
    """
    bg = np.asarray(background_tabular, dtype=np.float64)

    if explainer_name == "tabular_ablation":
        return {"background_data": bg, "n_draws": 8 if not fast else 3}
    if explainer_name == "occlusion":
        return {
            "background_data": bg,
            "time_window": 2,
            "perturbations_per_eval": 4 if not fast else 2,
        }
    if explainer_name == "feature_ablation":
        return {
            "background_data": bg,
            "group_mode": "none",
            "n_draws": 2 if not fast else 1,
        }
    if explainer_name == "shapley_sampling":
        return {"background_data": bg, "sample_size": 120 if not fast else 40}
    if explainer_name == "lime_tabular":
        return {"background_data": bg, "num_samples": 300 if not fast else 120}
    if explainer_name == "kernel_shap":
        return {
            "background_data": bg,
            "nsamples": 40 if not fast else 12,
            "background_k": 25,
        }
    if explainer_name == "sampling_shap":
        return {
            "background_data": bg,
            "nsamples": 120 if not fast else 30,
            "background_k": 25,
        }
    if explainer_name == "shap":
        return {
            "background_data": bg,
            "algorithm": "kernel",
            "nsamples": 40 if not fast else 12,
            "background_k": 25,
            "max_explain": int(batch_size),
        }
    raise ValueError(f"Unknown explainer name: {explainer_name}")


def build_explain_fn_single_step(
    tabular_explainer,
    *,
    explainer_name: str,
    background_tabular: np.ndarray,
) -> Callable:
    """Build explain_fn expected by FA-TS robustness metrics: (B,T,F) -> (B,T,F) with T=1."""
    bg = np.asarray(background_tabular, dtype=np.float64)

    def explain_fn(x_batch_btf: np.ndarray) -> np.ndarray:
        x_ts = np.asarray(x_batch_btf, dtype=np.float64)
        if x_ts.ndim != 3:
            raise ValueError(f"explain_fn expected (B,T,F), got {x_ts.shape}")
        if x_ts.shape[1] != 1:
            raise ValueError(f"explain_fn expects singleton T=1, got {x_ts.shape[1]}")
        x_tab = x_ts[:, 0, :]
        kwargs = get_explainer_kwargs(
            explainer_name,
            background_tabular=bg,
            batch_size=x_tab.shape[0],
            fast=True,
        )
        attr_tab = tabular_explainer.explain(
            x_tab,
            task="classification",
            return_abs=True,
            **kwargs,
        )
        attr_tab = np.asarray(attr_tab, dtype=np.float64)
        if attr_tab.shape != x_tab.shape:
            raise ValueError(
                f"tabular explainer output shape {attr_tab.shape}"
                f"does not match input {x_tab.shape}"
            )
        return attr_tab[:, np.newaxis, :]

    return explain_fn


# wrapper for scikit-learn style models


class SklearnPredictWrapper:
    """Simple adapter for sklearn classifiers to present ``predict_proba``.

    The wrapped object should implement ``predict_proba`` and optionally have a
    ``classes_`` attribute.
    """

    def __init__(self, model: BaseEstimatorProtocol):
        self.model = model
        maybe_classes = getattr(model, "classes_", None)
        self.classes_ = (
            np.asarray(maybe_classes, dtype=int) if maybe_classes is not None else None
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x), dtype=np.float64)


# new evaluation function accepting pretrained sklearn-like estimator + data


def evaluate_tabular_model(
    model: BaseEstimatorProtocol,
    x_tab: np.ndarray,
    bg_tab: np.ndarray,
    explainer_names: list[str] | None = None,
    metric_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run FA-TS metrics on a pretrained tabular model.

    Parameters
    ----------
    model
        A fitted sklearn-style estimator supporting ``predict_proba``.
    x_tab
        Input examples of shape ``(B, F)`` to score/explain.
    bg_tab
        Background dataset of shape ``(N, F)`` for explainers that need it.
    explainer_names
        List of explainer identifiers to evaluate.  Defaults to the same set used
        in :func:`main`.
    metric_ids
        List of metric identifiers to compute.  By default, all FA-TS
        multivariate time-series metrics for feature-attributions are used.

    Returns
    -------
    dict
        Contains two keys ``"sample_rows"`` and ``"summary_rows"`` whose values
        mirror the rows that would previously have been written to CSV files.
    """

    x_tab = np.asarray(x_tab, dtype=np.float64)
    bg_tab = np.asarray(bg_tab, dtype=np.float64)

    if explainer_names is None:
        explainer_names = [
            "tabular_ablation",
            "occlusion",
            "feature_ablation",
            "shapley_sampling",
            "lime_tabular",
            "kernel_shap",
            "sampling_shap",
            "shap",
        ]

    if metric_ids is None:
        metric_ids = list_metrics(
            explanation_type=ExplanationType.FEATURE_ATTRIBUTION,
            data_type=DataType.TIMESERIES_MULTIVARIATE,
        )

    model_name = type(model).__name__
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    # wrap the sklearn model so it looks like the TorchPredictWrapper used before
    predictor = SklearnPredictWrapper(model)
    ts_model = TabularAsSingleStepTSModel(predictor)

    print("Input tabular shape:", x_tab.shape)
    print("Background tabular shape:", bg_tab.shape)
    print("-" * 84)

    for explainer_name in explainer_names:
        print(f"[Explainer={explainer_name}]")
        explainer = ExplainerFactory.get(explainer_name, predictor)

        try:
            attr_kwargs = get_explainer_kwargs(
                explainer_name,
                background_tabular=bg_tab,
                batch_size=x_tab.shape[0],
                fast=False,
            )
            attr_tab = explainer.explain(
                x_tab,
                task="classification",
                return_abs=True,
                **attr_kwargs,
            )
            attr_tab = np.asarray(attr_tab, dtype=np.float64)
            if attr_tab.shape != x_tab.shape:
                raise ValueError(
                    f"explainer output shape {attr_tab.shape} does not match x shape {x_tab.shape}"
                )
            x_ts = to_single_step_btf(x_tab)
            attr_ts = to_single_step_btf(attr_tab)
            explain_fn_ts = build_explain_fn_single_step(
                explainer,
                explainer_name=explainer_name,
                background_tabular=bg_tab,
            )
        except Exception as exc:
            err = str(exc)
            print(f"  explainer failed: {err}")
            for metric_id in metric_ids:
                summary_rows.append(
                    {
                        "model": model_name,
                        "explainer": explainer_name,
                        "metric": metric_id,
                        "status": "error",
                        "n_scores": 0,
                        "mean": "",
                        "std": "",
                        "min": "",
                        "max": "",
                        "nan_count": "",
                        "inf_count": "",
                        "error": err,
                    }
                )
            continue

        for metric_id in metric_ids:
            metric = get_metric(metric_id, **METRIC_PARAMS.get(metric_id, {}))
            eval_kwargs: dict[str, Any] = {}
            if metric_id in NEEDS_EXPLAIN_FN:
                eval_kwargs["explain_fn"] = explain_fn_ts

            try:
                scores = metric.evaluate(
                    model=ts_model,
                    x=x_ts,
                    explanation=attr_ts,
                    **eval_kwargs,
                )
                scores_arr = np.asarray(scores, dtype=np.float64).ravel()
                nan_count = int(np.isnan(scores_arr).sum())
                inf_count = int(np.isinf(scores_arr).sum())

                for idx, s in enumerate(scores_arr):
                    sample_rows.append(
                        {
                            "model": model_name,
                            "explainer": explainer_name,
                            "metric": metric_id,
                            "sample_idx": idx,
                            "score": s,
                            "status": "ok",
                            "error": "",
                        }
                    )

                summary_rows.append(
                    {
                        "model": model_name,
                        "explainer": explainer_name,
                        "metric": metric_id,
                        "status": "ok",
                        "n_scores": int(scores_arr.size),
                        "mean": float(np.nanmean(scores_arr))
                        if scores_arr.size
                        else "",
                        "std": float(np.nanstd(scores_arr)) if scores_arr.size else "",
                        "min": float(np.nanmin(scores_arr)) if scores_arr.size else "",
                        "max": float(np.nanmax(scores_arr)) if scores_arr.size else "",
                        "nan_count": nan_count,
                        "inf_count": inf_count,
                        "error": "",
                    }
                )

                mean_txt = (
                    float(np.nanmean(scores_arr)) if scores_arr.size else float("nan")
                )
                print(
                    f"  {metric_id:35s} "
                    f"mean={mean_txt:.6f} nan={nan_count} inf={inf_count}"
                )
            except Exception as exc:
                err = str(exc)
                print(f"  {metric_id:35s} FAILED: {err}")
                summary_rows.append(
                    {
                        "model": model_name,
                        "explainer": explainer_name,
                        "metric": metric_id,
                        "status": "error",
                        "n_scores": 0,
                        "mean": "",
                        "std": "",
                        "min": "",
                        "max": "",
                        "nan_count": "",
                        "inf_count": "",
                        "error": err,
                    }
                )
                sample_rows.append(
                    {
                        "model": model_name,
                        "explainer": explainer_name,
                        "metric": metric_id,
                        "sample_idx": "",
                        "score": "",
                        "status": "error",
                        "error": err,
                    }
                )

    return sample_rows, summary_rows


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Create dummy dataset
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=8, n_redundant=2, random_state=42
    )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train sklearn model
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # Run evaluation
    _, summary_rows = evaluate_tabular_model(
        model=model,
        x_tab=X_test[:10],  # Use first 10 test samples
        bg_tab=X_train[:50],  # Use first 50 training samples as background
    )
    trustworthiness_metrics = pd.DataFrame(summary_rows)
    ranking_all, top_k_aggregated_metrics, _ = rank_and_select_top_k(
        trustworthiness_metrics, coeffs={"F": 1.0, "R": 1.0, "C": 1.0, "P": 1.0}, k=5
    )

    values = (
        top_k_aggregated_metrics[
            ["explainer", "Faithfulness", "Robustness", "Complexity"]
        ]
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    metrics = []
    for row in values:
        explainer = row.pop("explainer", None)
        metrics.extend(
            {metric_name: {"score": score, "description": explainer}}
            for metric_name, score in row.items()
        )

    print(metrics)
