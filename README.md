# aisc-plugin-explitest

A Python plugin for evaluating the explainability of ML model predictions using the [ExpliTest](https://github.com/serval-uni-lu/ExpliTest) framework. Built on the [aisc-plugin-interface](https://github.com/lux-ai-factory/aisc-plugin-interface) framework.

## Features

- **XAI Evaluation** — measures **Faithfulness**, **Robustness**, **Complexity**, and **Performance** of model explanations across multiple explainers
- **Multiple Explainers** — supports `tabular_ablation`, `feature_ablation`, `lime_tabular`, and `shap`
- **Automatic Config Parsing** — auto-detects feature types (Integer, Float, Categorical, Date) from the dataset
- **Default Credit Scoring Config** — bundled configuration for Lending Club credit scoring data
- **Time-Windowed Evaluation** — compute metrics over sliding windows with configurable frequency
- **Radar Chart Visualization** — aggregated XAI metric overview via radar chart
- **ONNX Model Support** — loads and runs ONNX models with automatic probability output detection
- **Multiple Input Formats** — CSV and Parquet datasets, ONNX models

## Installation

```bash
git clone git@github.com:lux-ai-factory/aisc-plugin-explitest.git
cd aisc-plugin-explitest
uv sync
```

## Quick Start

The plugin takes a reference dataset (training data), an evaluated dataset (test data), and an ONNX model.

```python
from explitest_plugin import ExpliTestPlugin

plugin = ExpliTestPlugin()
plugin.set_input_content("train-dataset", train_data_bytes)
plugin.set_input_content("test-dataset", test_data_bytes)
plugin.set_input_content("model", model_bytes)

results = plugin.evaluate({
    "features": [...],
    "target_feature": "charged_off",
    "date_feature": "issue_d",       # optional
    "frequency": "60D",               # optional
    "window_size": "120 days",        # optional
})
```

## How It Works

The plugin bridges tabular ML models to ExpliTest's multivariate time-series evaluation framework by adding a singleton time axis `(B, F) → (B, 1, F)`. Each explainer generates feature attributions, which are then evaluated across four metric categories:

### XAI Metrics

| Category | Description |
|----------|-------------|
| **Faithfulness** | Correlation between attributions and model effect; pixel-flipping score drop |
| **Robustness** | Attribution sensitivity to small input perturbations; continuity for nearby inputs |
| **Complexity** | Sparseness and entropy of attribution distributions |
| **Performance** | Model F1-score on the test set |

### Explainers

| Explainer | Description |
|-----------|-------------|
| **Tabular Ablation** | Ablates features one by one to measure impact |
| **Feature Ablation** | Feature ablation with configurable grouping |
| **LIME Tabular** | Local Interpretable Model-agnostic Explanations |
| **SHAP** | SHapley Additive exPlanations (kernel approximation) |

### All ExpliTest Metrics

This plugin uses the **feature-attribution** metrics (FA-TS) for evaluation. The full ExpliTest library also includes **counterfactual** metrics. Below is the complete catalog.

#### Robustness

| Metric ID | What It Measures |
|-----------|------------------|
| `cf_validity_under_noise_tabular` | Fraction of perturbed CFs that stay valid |
| `cf_validity_margin_tabular` | Signed distance to target boundary |
| `cf_local_lipschitz_at_cf_tabular` | Model sensitivity around the counterfactual |
| `cf_regeneration_stability_tabular` | CF stability when the input is slightly perturbed |
| `fa_ts_mv_avg_sensitivity` | Attribution sensitivity to small input perturbations |
| `fa_ts_mv_continuity` | Attribution continuity for nearby inputs |

#### Faithfulness

| Metric ID | What It Measures |
|-----------|------------------|
| `cf_path_auc_distance_tabular` | Area under distance-to-target curve along interpolation path |
| `cf_path_monotonicity_tabular` | Whether prediction improves monotonically along path |
| `cf_directional_alignment_tabular` | Alignment between geometric and prediction progress |
| `cf_endpoint_target_error_tabular` | Prediction error at the counterfactual endpoint |
| `fa_ts_mv_faithfulness_correlation` | Correlation between attribution and model effect |
| `fa_ts_mv_pixel_flipping` | Score drop under top-attribution perturbation |

#### Complexity

| Metric ID | What It Measures |
|-----------|------------------|
| `cf_delta_sparsity_tabular` | Fraction of features changed |
| `cf_delta_proximity_tabular` | Distance between original and CF (L1, L2, L∞, Mahalanobis) |
| `cf_delta_entropy_tabular` | Entropy of the change distribution |
| `cf_delta_gini_tabular` | Concentration of changes (Gini coefficient) |
| `cf_constraint_violations_tabular` | Number of violated feasibility / immutability constraints |
| `fa_ts_mv_sparseness_element` | Sparseness at element level |
| `fa_ts_mv_sparseness_channel` | Sparseness aggregated by channel |
| `fa_ts_mv_complexity_entropy_element` | Entropy-based complexity at element level |
| `fa_ts_mv_complexity_entropy_channel` | Entropy-based complexity aggregated by channel |

### Explainer Ranking

Per-explainer scores are normalized per metric and aggregated into a weighted final score:

```
final_score = T × (F×Faith + R×Rob + C×Comp) + P×Perf
```

Where `F`, `R`, `C`, `P` are configurable coefficients (defaults: all 1.0). The top-k explainers are selected based on this ranking.

## Time-Windowed Evaluation

Metrics can be computed over the entire dataset or over temporal windows:

```python
config = {
    "date_feature": "issue_d",     # Column containing dates
    "frequency": "60D",             # Window hop size
    "window_size": "120 days",      # Window duration
}
```

## Development

### Setup

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Commands

```bash
# Linting
uv run ruff check src/

# Linting with auto-fix
uv run ruff check --fix src/

# Type checking
uv run ty check src/

# Format code
uv run ruff format src/

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run checks before each commit:

- **Ruff** — Linting and formatting
- **ty** — Type checking

Hooks are installed automatically when you run `uv run pre-commit install`. To skip hooks temporarily:

```bash
git commit --no-verify -m "message"
```

### Project Structure

```
src/explitest_plugin/
├── __init__.py              # Public exports (ExpliTestPlugin)
├── plugin.py                # ExpliTestPlugin — main evaluation logic
├── config_form.py           # Pydantic config + UI schema
├── trustworthiness_xai.py   # FA-TS metric evaluation + explainer ranking
├── data_input_provider.py   # CSV/Parquet data reader + date iterator
├── model_input_provider.py  # ONNX model loader + predictor adapter
├── iterators.py             # Date windowing utilities
├── utils.py                 # Feature model, metric grouping, decorators
└── credit_scoring_config.json  # Default config for Lending Club data
```

## Tech Stack

- **Python** 3.12+
- **aisc-plugin-interface** — Plugin framework
- **ExpliTest** — XAI evaluation library
- **NumPy/pandas** — Data processing
- **SciPy** — Statistical utilities
- **scikit-learn** — Train/test splitting, F1 scoring
- **ONNX Runtime** — Model inference
- **LIME** — Local interpretable explanations
- **SHAP** — Shapley value explanations
- **explainers** — Explainer abstraction layer
- **uv** — Package management
- **Ruff** — Linting and formatting
- **ty** — Type checking

## Plugin Metadata

### ExpliTest Plugin

| Field | Value |
|-------|-------|
| Name | Explainability |
| Description | Evaluates explainability of ML model predictions using ExpliTest metrics (Faithfulness, Robustness, Complexity, Performance) across multiple explainers (Ablation, LIME, SHAP) with time-windowed evaluation and radar chart visualization. |
| License | - |
| Verification type | Technical test |
| Project | [aisc-plugin-explitest](https://github.com/lux-ai-factory/aisc-plugin-explitest) |
| Branch | main |
| Version | 0.2.2 |
| Project maturity | Deployed |
| Scientific reference | [ExpliTest](https://github.com/serval-uni-lu/ExpliTest) |
| Verification targets | [Explainability] [XAI Evaluation] [Trustworthiness] |
| Sector | [AI/ML] [Data Science] [MLOps] |
