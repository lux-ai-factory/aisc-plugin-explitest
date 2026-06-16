import copy
import json
from pathlib import Path
from datetime import datetime
from typing import Any, TypeVar

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    PluginFeatureFlags,
    evaluation_input,
    InputType,
    MetricVisualization,
    ChartType,
)

from .utils import add_metrics, group_metrics, Feature, FeatureType
from .config_form import ConfigForm, FORM_UI_SCHEMA
from .data_input_provider import DataFrameProvider, dataframe_iter
from .model_input_provider import OnnxInputProvider, OnnxModelSession

T = TypeVar("T")

DEFAULT_CONFIG = (
    Path(__file__).parent.resolve().absolute() / "credit_scoring_config.json"
)


@add_metrics
@evaluation_input(
    name="test-dataset",
    label="Test Dataset",
    input_provider_class=DataFrameProvider,
    input_type=InputType.DATASET,
    required=True,
)
@evaluation_input(
    name="train-dataset",
    label="Train Dataset",
    input_provider_class=DataFrameProvider,
    input_type=InputType.DATASET,
    required=True,
)
@evaluation_input(
    name="model",
    label="Model",
    input_provider_class=OnnxInputProvider,
    input_type=InputType.MODEL,
    required=True,
)
class ExpliTestPlugin(BaseEvaluationPlugin[ConfigForm]):
    plugin_name = "Explainability"

    ui_icon = "search_insights"

    form_ui_schema = FORM_UI_SCHEMA

    xai_metric_names = [
        "Faithfulness",
        "Robustness",
        "Complexity",
        "Performance",
    ]

    explainer_names = [
        "tabular_ablation",
        "feature_ablation",
        # "shapley_sampling",
        "lime_tabular",
        # "kernel_shap",
        # "sampling_shap",
        "shap",
    ]

    @classmethod
    def metric_names(cls) -> list[str]:
        return cls.xai_metric_names

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(can_parse_config_from_dataset=True)

    def parse_config_from_dataset(self, file_content: bytes) -> dict | None:
        import pandas as pd

        self.logger.info("Parsing config from dataset")

        config: ConfigForm = ConfigForm(
            target_feature=None,
            date_feature=None,
            frequency="",
            window_size="",
            features=[],
        )

        try:
            input_provider = DataFrameProvider(file_content)
            input_provider.get_data()
            df: pd.DataFrame = input_provider.get_data()
        except Exception:
            self.logger.exception("Failed to load dataset for config parsing")
            raise

        if df.empty:
            self.logger.warning("Dataset is empty, returning default config")
            return config.model_dump()

        self.logger.debug(
            "Dataset loaded with %d rows and %d columns", len(df), len(df.columns)
        )

        for col_name in df.columns:
            col_data = df[col_name]

            feature_type = FeatureType.CATEGORICAL

            # Check for Date
            if pd.api.types.is_datetime64_any_dtype(col_data):
                feature_type = FeatureType.DATE
            elif pd.api.types.is_object_dtype(col_data):
                temp = pd.to_datetime(col_data, errors="coerce")
                if temp.isna().any():
                    self.logger.warning(
                        "Attempted to parse '%s' as a date, but failed", col_name
                    )
                else:
                    feature_type = FeatureType.DATE

            # Check for Numeric
            if feature_type != FeatureType.DATE:
                if pd.api.types.is_integer_dtype(col_data):
                    feature_type = FeatureType.INTEGER
                elif pd.api.types.is_float_dtype(col_data):
                    feature_type = FeatureType.FLOAT

            # Get Min/Max for Numeric types
            if feature_type in [FeatureType.INTEGER, FeatureType.FLOAT]:
                col_min = float(col_data.min()) if not pd.isna(col_data.min()) else 0.0
                col_max = float(col_data.max()) if not pd.isna(col_data.max()) else 0.0
            else:
                # For Categorical or Date, min/max usually aren't numeric ranges
                col_min = 0.0
                col_max = 0.0

            feature: Feature = Feature(
                name=col_name, min=col_min, max=col_max, type=feature_type
            )
            config.features.append(feature)
            self.logger.debug(
                "Detected feature '%s' as %s (min=%.2f, max=%.2f)",
                col_name,
                feature_type,
                col_min,
                col_max,
            )

        self.logger.info("Parsed %d features from dataset", len(config.features))

        # Use the default config only if all its feature names are present in the dataset
        try:
            with open(DEFAULT_CONFIG, "r") as f:
                data = json.load(f)
            default_form = ConfigForm(**data)
            default_feature_names = {f.name for f in default_form.features}
            if default_feature_names.issubset({f.name for f in config.features}):
                self.logger.info("Default config matches dataset, using it")
                return default_form.model_dump()
        except Exception:
            pass

        return config.model_dump()

    def on_config_change(
        self, form_data: ConfigForm | None
    ) -> tuple[ConfigForm | None, dict[str, Any], dict[str, Any]]:
        config_schema, ui_schema = self.get_full_schema()
        ui_schema = copy.deepcopy(ui_schema)

        if form_data is None:
            ui_schema["date_feature"] = {"ui:widget": "hidden"}
            ui_schema["target_feature"] = {"ui:widget": "hidden"}
            ui_schema["frequency"] = {"ui:widget": "hidden"}
            ui_schema["window_size"] = {"ui:widget": "hidden"}
            return None, config_schema, ui_schema

        # Convert to dict for property access if needed
        form_dict = (
            form_data.model_dump() if isinstance(form_data, ConfigForm) else form_data
        )

        if (
            "properties" in config_schema
            and "date_feature" in config_schema["properties"]
        ):
            possible_date_features = [
                f["name"]
                for f in form_dict.get("features", [])
                if f["type"] in (FeatureType.DATE, FeatureType.CATEGORICAL)
            ]
            if possible_date_features:
                # NOTE: adding an empty string will force the user to make a choice
                possible_date_features.insert(0, "")
                config_schema["properties"]["date_feature"] = {
                    "title": "Date Feature",
                    "type": "string",
                    "enum": possible_date_features,
                    "default": possible_date_features[0],
                }
                if isinstance(form_data, ConfigForm):
                    if form_data.date_feature is None:
                        form_data = form_data.model_copy(
                            update={"date_feature": possible_date_features[0]}
                        )
                elif isinstance(form_data, dict):
                    if form_data.get("date_feature") is None:
                        form_data = {
                            **form_data,
                            "date_feature": possible_date_features[0],
                        }
            else:
                ui_schema["date_feature"] = {"ui:widget": "hidden"}
                ui_schema["frequency"] = {"ui:widget": "hidden"}
                ui_schema["window_size"] = {"ui:widget": "hidden"}

        if (
            "properties" in config_schema
            and "target_feature" in config_schema["properties"]
        ):
            possible_target_features = [
                f["name"]
                for f in form_dict.get("features", [])
                if f["type"]
                in (FeatureType.INTEGER, FeatureType.FLOAT, FeatureType.CATEGORICAL)
            ]
            if possible_target_features:
                # NOTE: adding an empty string will force the user to make a choice
                possible_target_features.insert(0, "")
                config_schema["properties"]["target_feature"] = {
                    "title": "Target Feature",
                    "type": "string",
                    "enum": possible_target_features,
                    "default": possible_target_features[-1],
                }
                if isinstance(form_data, ConfigForm):
                    if form_data.target_feature is None:
                        form_data = form_data.model_copy(
                            update={"target_feature": possible_target_features[-1]}
                        )
                elif isinstance(form_data, dict):
                    if form_data.get("target_feature") is None:
                        form_data = {
                            **form_data,
                            "target_feature": possible_target_features[-1],
                        }
            else:
                ui_schema["target_feature"] = {"ui:widget": "hidden"}

        return form_data, config_schema, ui_schema

    def compute_performance(self, x_test, y_true, model, date=None):
        import numpy as np
        from sklearn.metrics import f1_score

        if date is None:
            date = datetime.now()

        y_pred = np.argmax(model.predict(x_test, probabilities=True), axis=1)
        per_score = f1_score(y_true, y_pred, zero_division=0)

        return [
            {
                "Performance": {
                    "score": per_score,
                    "time": date,
                    "description": explainer,
                }
            }
            for explainer in self.explainer_names
        ]

    def xai_metrics(self, x_train, x_test, y_true, model, date=None):
        import numpy as np
        import pandas as pd
        from .trustworthiness_xai import (
            evaluate_tabular_model,
            rank_and_select_top_k,
            _infer_metric_category,
            EXPLAINER_NAMES,
        )

        if date is None:
            date = datetime.now()

        it_explainer_names = self.progress_bar(
            EXPLAINER_NAMES,
            desc="Evaluating each Explainer",
            start=1,
            total=len(EXPLAINER_NAMES),
        )

        _, summary_rows = evaluate_tabular_model(
            model=model,
            bg_tab=x_train,
            x_tab=x_test,
            explainer_names=it_explainer_names,  # ty: ignore[invalid-argument-type]
        )

        explainability_metrics = pd.DataFrame(summary_rows)

        # save xai metrics as artifact
        artifact = explainability_metrics.copy()
        artifact.insert(
            artifact.columns.get_loc("metric") + 1,
            "metric_category",
            artifact["metric"].map(_infer_metric_category),
        )

        self.upload_artifact(
            "xai_metrics.csv",
            artifact.to_csv(index=False).encode("utf-8"),
        )
        del artifact

        _, top_k_aggregated_metrics, _ = rank_and_select_top_k(
            explainability_metrics,
            coeffs={"F": 1.0, "R": 1.0, "C": 1.0, "P": 1.0},
            k=100,
        )

        values = (
            top_k_aggregated_metrics[["explainer", *self.xai_metric_names[:-1]]]
            .replace({np.nan: None})
            .to_dict(orient="records")
        )

        metrics = []
        for row in values:
            explainer = row.pop("explainer", None)
            metrics.extend(
                {metric_name: {"score": score, "description": explainer, "time": date}}
                for metric_name, score in row.items()
                if explainer in self.explainer_names
            )

        return metrics

    def evaluate(self, config_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        import pandas as pd
        from onnxruntime import InferenceSession
        from sklearn.model_selection import train_test_split

        config = self.validate_config_form_data(config_data)

        target_col = config.target_feature
        date_feature = config.date_feature
        frequency = config.frequency
        window_size = config.window_size

        features = []
        columns_features = []

        for feature in config.features:
            if feature.name not in (target_col, date_feature):
                features.append(feature)
                columns_features.append(feature.name)

        try:
            df_train = self.get_input_data("train-dataset")
        except Exception:
            self.logger.exception("Failed to load train dataset")
            raise
        assert isinstance(df_train, pd.DataFrame)

        try:
            df_test = self.get_input_data("test-dataset")
        except Exception:
            self.logger.exception("Failed to load test dataset")
            raise
        assert isinstance(df_test, pd.DataFrame)

        x_train = df_train[columns_features]
        x_test = df_test[columns_features]
        y_true = df_test[target_col].to_numpy()

        dates_masks = list(
            dataframe_iter(df_test, date_feature, frequency, window_size)
        )
        default_date = dates_masks[-1][0] or datetime.now()

        session = self.get_input_data("model")
        assert isinstance(session, InferenceSession)
        model_session = OnnxModelSession(session)

        # Performance metrics
        metrics = self.compute_performance(
            x_test, y_true, model_session, date=default_date
        )

        # NOTE: only a subset of the test set is considered
        subset_size = min(
            max(int(0.0001 * len(y_true)), 100),
            len(y_true) - len(set(y_true)),
        )

        x_test, _, y_true, _ = train_test_split(
            x_test,
            y_true,
            train_size=subset_size,
            stratify=y_true,
            random_state=42,
        )

        # XAI
        metrics.extend(
            self.xai_metrics(
                x_train[:50],
                x_test,
                y_true,
                model_session,
                date=default_date,
            )
        )

        return group_metrics(metrics)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        return [
            MetricVisualization(
                chart_type=ChartType.RADAR,
                metrics=self.xai_metric_names,
            ),
        ]
