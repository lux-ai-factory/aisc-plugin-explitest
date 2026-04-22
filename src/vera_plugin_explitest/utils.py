import uuid
from datetime import datetime
from enum import Enum
from collections import defaultdict
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field, field_serializer

from vera_plugin_interface import Measure, metric


class _HasMetricNames(Protocol):
    @classmethod
    def metric_names(cls) -> list[str]: ...


T = TypeVar("T", bound=_HasMetricNames)


class FeatureType(str, Enum):
    INTEGER = "Integer"
    FLOAT = "Float"
    CATEGORICAL = "Categorical"
    DATE = "Date"


class Feature(BaseModel):
    pid: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(...)
    min: float = Field(...)
    max: float = Field(...)
    type: FeatureType = Field(...)

    @field_serializer("pid")
    def serialize_pid(self, pid: uuid.UUID | None) -> str | None:
        return str(pid) if pid is not None else None


def group_metrics(
    dicts: list[dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Group a list of metric dictionaries by metric name into a single dictionary.

    Args:
        dicts (list[dict]): A list of dictionaries to group.

    Returns:
        dict: A grouped dictionary with consolidated values.

    Example:
        >>> dicts = [{"a": {"x": 1}}, {"a": {"x": 2, "y": 3}, "b": {"z": 4}}]
        >>> result = group_metrics(dicts)
        >>> print(result)
        {'a': [{'x': 1}, {'x': 2, 'y': 3}], 'b': [{'z': 4}]}
    """
    grouped = defaultdict(list)

    for item in dicts:
        for key, value in item.items():
            grouped[key].append(value)

    return dict(grouped)


def add_metrics(cls: type[T]) -> type[T]:
    """Class decorator that automatically adds @metric decorated export methods."""
    for name in cls.metric_names():

        @metric(name)
        def fct(self, evaluation_output: dict, _name=name) -> list[Measure]:
            measures: list[dict] = evaluation_output.get(_name, [])

            return [
                Measure(
                    name=_name,
                    score=float(score),
                    time=measure.get("time", datetime.now()),
                    description=measure.get("description"),
                )
                for measure in measures
                if (score := measure["score"]) is not None
            ]

        setattr(cls, f"export_metric_{name}", fct)

    return cls
