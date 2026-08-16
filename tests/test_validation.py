import pandas as pd
import pytest
import os
from src.data.validation import (
    validate_schema,
    validate_no_missing_values,
    validate_no_duplicates,
    validate_target_range,
    run_validation,
)

# Sanity Check against actual dataset
@pytest.mark.skipif(not os.path.exists("data/raw/day.csv"), reason="Full dataset not available in CI")
def test_run_validation_on_real_data():
    df = run_validation("data/raw/day.csv")
    assert len(df) == 731
    assert "cnt" in df.columns

def test_validate_scheme_raises_on_missing_column():
    df = pd.DataFrame({"instant": [1], "dteday": ["2021-01-01"]})
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_schema(df)

def test_validate_no_missing_values_raises_on_nulls():
    df = pd.DataFrame({"cnt": [1, None, 3]})
    with pytest.raises(ValueError, match="Found missing values"):
        validate_no_missing_values(df)


def test_validate_no_duplicates_raises_on_duplicate_dates():
    df = pd.DataFrame({"dteday": ["2011-01-01", "2011-01-01"]})
    with pytest.raises(ValueError, match="duplicate data rows"):
        validate_no_duplicates(df)


def test_validate_target_range_raises_on_negative_values():
    df = pd.DataFrame({"cnt": [10, -5, 20]})
    with pytest.raises(ValueError, match="negative values"):
        validate_target_range(df)       