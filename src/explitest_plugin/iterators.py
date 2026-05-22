from datetime import datetime

import pandas as pd


def get_date_batches(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    freq: str | None,
    window: str | None,
    date_round: str | None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate a list of date batches based on specified parameters.

    Args:
        start_date (pd.Timestamp): The start date of the overall range
        end_date (pd.Timestamp): The end date of the overall range
        freq (str): The frequency of batch starts (e.g., '1D' for daily)
        window (str): The size of each window (e.g., '7D' for 7 days)
        date_round (str): How to round the dates (e.g., 'D' for day, 'H' for hour)

    Returns:
        list[tuple[pd.Timestamp, pd.Timestamp]]: List of (start, end) timestamp pairs
    """
    # Special case: if start_date equals end_date, create a single batch
    if start_date == end_date:
        # Create a batch that includes all data from that single date
        batch_start = start_date
        batch_end = start_date + pd.Timedelta(
            days=1
        )  # Next day to include all data from start_date
        assert isinstance(batch_end, pd.Timestamp)
        return [(batch_start, batch_end)]

    # Round dates if specified
    start_date_round = start_date
    end_date_round = end_date
    if date_round:
        start_date_round = start_date.floor(date_round)
        end_date_round = end_date.ceil(date_round)

    # Generate the date ranges at specified frequency
    date_ranges = pd.date_range(start_date_round, end_date_round, freq=freq)
    # Calculate window ends by adding the window size to each start date
    windows = date_ranges.to_series().add(pd.Timedelta(window))

    # Filter out windows that extend beyond the end date
    valid_batches = windows[windows <= end_date_round]

    return list(zip(date_ranges[: len(valid_batches)], valid_batches))


class DateIterator:
    """Iterator for processing dataframes in temporal batches.

    This class provides functionality to iterate over a DataFrame in time-based windows,
    useful for temporal analysis and time-series processing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_feature: str | None,
        freq: str | None,
        window: str | None,
        date_round: str | None = "1 D",
    ) -> None:
        """Initialize the DateIterator.

        Args:
            df (pd.DataFrame): The DataFrame to iterate over
            date_feature (str): The column name containing dates
            freq (str): The frequency of batch starts (e.g., '1D' for daily)
            window (str): The size of each window (e.g., '7D' for 7 days)
            date_round (str): How to round the dates (e.g., 'D' for day)
        """
        self.date_feature = date_feature

        if self.date_feature is None:
            self.df_date_series = df.index
        else:
            self.df_date_series = pd.to_datetime(df[self.date_feature])

        self.start_date = self.df_date_series.min()
        self.end_date = self.df_date_series.max()

        self.date_round = date_round or None
        self.window = window or None
        self.freq = freq or None

        if None in (self.date_feature, self.date_round, self.window, self.freq):
            offset = 1 if self.date_feature is None else pd.Timedelta(days=1)
            self.batches = [(self.start_date, self.end_date + offset)]
        else:
            self.batches = get_date_batches(
                self.start_date, self.end_date, freq, window, date_round
            )

        self.index = 0

    def __iter__(self) -> "DateIterator":
        """Return the iterator object."""
        return self

    def __next__(self) -> tuple[datetime | None, pd.Series]:
        """Get the next batch of data.

        Returns:
            A tuple of (date, mask) where mask is a boolean Series for the current batch.

        Raises:
            StopIteration: When there are no more batches to process.
        """
        if self.index >= len(self.batches):
            raise StopIteration

        start, end = self.batches[self.index]
        self.index += 1

        mask = (self.df_date_series >= start) & (self.df_date_series < end)

        if not mask.any():
            return self.__next__()

        # Filter DataFrame for current time window
        date = (
            None
            if self.date_feature is None
            else self.df_date_series[mask].max().to_pydatetime()
        )
        return date, mask
