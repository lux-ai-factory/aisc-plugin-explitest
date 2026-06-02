import io
from collections.abc import Iterator
from datetime import datetime

import pandas as pd
from aisc_plugin_interface import BaseInputProvider

from .iterators import DateIterator


class DataFrameProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> pd.DataFrame:
        import pandas as pd

        file_stream = io.BytesIO(file_content)
        try:
            return pd.read_parquet(file_stream)
        except Exception:
            file_stream.seek(0)
            try:
                return pd.read_csv(file_stream)
            except Exception as e:
                raise ValueError("File is neither a valid Parquet nor CSV.") from e


def dataframe_iter(
    dataframe: pd.DataFrame,
    date_feature: str | None,
    frequency: str | None,
    window_size: str | None,
    date_round: str | None = "1 D",
) -> Iterator[tuple[datetime | None, pd.Series]]:
    if date_feature is not None:
        dataframe[date_feature] = pd.to_datetime(dataframe[date_feature])

    yield from DateIterator(dataframe, date_feature, frequency, window_size, date_round)
