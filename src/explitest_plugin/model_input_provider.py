from typing import TYPE_CHECKING, Any

from vera_plugin_interface import BaseInputProvider

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    from onnxruntime import InferenceSession


class OnnxInputProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> "InferenceSession":
        from onnxruntime import InferenceSession

        session: InferenceSession = InferenceSession(file_content)
        return session


class OnnxModelSession:
    def __init__(self, session: "InferenceSession"):
        self.session = session

    def predict(
        self, x_test: "npt.ArrayLike | pd.DataFrame", probabilities: bool = False
    ) -> "npt.NDArray[np.float32]":
        import pandas as pd
        import numpy as np

        x_arr: "npt.NDArray[Any]"
        if isinstance(x_test, pd.DataFrame):
            x_arr = x_test.to_numpy()
        elif isinstance(x_test, np.ndarray):
            x_arr = x_test
        else:
            raise ValueError(f"x_test should be np.ndarray, found: {type(x_test)}")

        if self.session is None:
            raise ValueError(
                f"Please call {self.__class__.__name__}._read_data(...) first"
            )

        # determine expected dtype
        expected = self.session.get_inputs()[0].type
        target_dtype: npt.DTypeLike
        if "tensor(float)" in expected:
            target_dtype = np.float32
        elif "tensor(double)" in expected:
            target_dtype = np.float64
        else:
            target_dtype = np.float32  # default fallback

        x = np.ascontiguousarray(
            x_arr.astype(target_dtype)  # ty: ignore[no-matching-overload]
        )
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Get output candidates
        out_candidates = self.session.get_outputs()

        if probabilities:
            # prefer a 'prob'/'probab' output if available (for classification)
            idx = next(
                (i for i, o in enumerate(out_candidates) if "prob" in o.name.lower()),
                None,
            )
            if idx is None:
                # fallback: second output if it exists, else first
                idx = int(len(out_candidates) >= 2)
        else:
            # for regression, use first output
            idx = 0
        output_name = out_candidates[idx].name

        raw = self.session.run([output_name], {self.session.get_inputs()[0].name: x})[0]

        # handle ZipMap case for classification probabilities (list of dicts)
        if probabilities and isinstance(raw, list) and raw and isinstance(raw[0], dict):
            keys = list(raw[0].keys())
            arr = np.array(
                [
                    [row.get(k, 0.0) for k in keys]  # ty: ignore[unresolved-attribute]
                    for row in raw
                ],
                dtype=np.float32,
            )
            return arr

        arr = np.array(raw)
        if probabilities and arr.ndim == 1:
            arr = arr[:, None]
        if not probabilities and arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        return arr.astype(np.float32)

    def predict_proba(
        self, x_test: "npt.ArrayLike | pd.DataFrame", probabilities: bool = False
    ) -> "npt.NDArray[np.float32]":
        return self.predict(x_test, probabilities=True)
