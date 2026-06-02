from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aisc_plugin_interface import BaseInputProvider

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    from onnxruntime import InferenceSession


type Array = np.ndarray
type ArrayLike = npt.ArrayLike | pd.DataFrame
type FloatArray = npt.NDArray[np.float32]


class OnnxInputProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> "InferenceSession":
        from onnxruntime import InferenceSession

        session: InferenceSession = InferenceSession(file_content)
        return session


@dataclass(slots=True, frozen=True)
class OnnxIOInfo:
    input_name: str
    input_dtype: "np.dtype[Any]"
    output_names: list[str]
    probability_output_index: int | None


class OnnxModelSession:
    __slots__ = ("session", "io")

    def __init__(self, session: "InferenceSession") -> None:
        self.session = session
        self.io = self._build_io_info()

    def predict(self, x: ArrayLike, probabilities: bool = False) -> FloatArray:
        return self.predict_proba(x) if probabilities else self._predict_labels(x)

    def _predict_labels(self, x: ArrayLike) -> FloatArray:
        import numpy as np

        if self.io.probability_output_index is not None:
            proba = self.predict_proba(x)
            return np.argmax(proba, axis=1).astype(np.float32)

        arr = np.asarray(self._run(x)[0])

        if arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        return arr.astype(np.float32, copy=False)

    def predict_proba(self, x: ArrayLike) -> FloatArray:
        import numpy as np

        outputs = self._run(x)

        idx = (
            self.io.probability_output_index
            if self.io.probability_output_index is not None
            else 0
        )

        raw = outputs[idx]

        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return self._zipmap_to_array(raw)

        arr = np.asarray(raw, dtype=np.float32)

        if arr.ndim == 1:
            arr = np.column_stack((1.0 - arr, arr))

        elif arr.ndim == 2 and arr.shape[1] == 1:
            arr = np.column_stack((1.0 - arr[:, 0], arr[:, 0]))

        if (
            arr.ndim == 2
            and arr.shape[1] == 2
            and np.allclose(arr.sum(axis=1), 0, atol=1e-4)
        ):
            # Binary output with zipmap=False: centered scores, column 1 = P(positive)
            prob_positive = np.clip(arr[:, 1:2], 0, 1)
            prob_negative = 1.0 - prob_positive
            arr = np.concatenate([prob_negative, prob_positive], axis=1)
        elif not self._is_probability_matrix(arr):
            arr = self._softmax(arr)

        return arr.astype(np.float32, copy=False)

    def _run(self, x: ArrayLike):
        arr = self._prepare_input(x)

        return self.session.run(
            self.io.output_names,
            {self.io.input_name: arr},
        )

    def _prepare_input(self, x: ArrayLike):
        import numpy as np
        import pandas as pd

        if isinstance(x, pd.DataFrame):
            arr = x.to_numpy(copy=False)
        elif isinstance(x, np.ndarray):
            arr = x
        else:
            raise ValueError(f"x_test should be np.ndarray, found: {type(x)}")

        if arr.ndim == 1:
            arr = np.reshape(arr, (1, -1))

        if arr.dtype != self.io.input_dtype:
            arr = np.ascontiguousarray(arr, dtype=self.io.input_dtype)
        else:
            arr = np.ascontiguousarray(arr)

        return arr

    def _build_io_info(self) -> OnnxIOInfo:
        input_meta = self.session.get_inputs()[0]
        outputs = self.session.get_outputs()

        probability_output_index = next(
            (
                i
                for i, out in enumerate(outputs)
                if any(k in out.name.lower() for k in ("prob", "proba", "probability"))
            ),
            1 if len(outputs) >= 2 else None,
        )

        return OnnxIOInfo(
            input_name=input_meta.name,
            input_dtype=self._onnx_dtype_to_numpy(input_meta.type),
            output_names=[o.name for o in outputs],
            probability_output_index=probability_output_index,
        )

    @staticmethod
    def _onnx_dtype_to_numpy(onnx_type: str):
        import numpy as np

        return {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }.get(onnx_type, np.float32)

    @staticmethod
    def _is_probability_matrix(arr: Array):
        import numpy as np

        return (
            arr.ndim == 2
            and np.all(arr >= 0.0)
            and np.all(arr <= 1.0)
            and np.allclose(arr.sum(axis=1), 1.0, atol=1e-4)
        )

    @staticmethod
    def _softmax(x: Array) -> Array:
        import numpy as np

        x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x / np.sum(x, axis=1, keepdims=True)

    @staticmethod
    def _zipmap_to_array(rows: list[dict[Any, Any]]) -> FloatArray:
        import numpy as np

        keys = sorted(rows[0])

        out = np.empty((len(rows), len(keys)), dtype=np.float32)

        for i, row in enumerate(rows):
            for j, key in enumerate(keys):
                out[i, j] = row.get(key, 0.0)

        return out
