from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast import forecast_demand


def test_model_files_exist():
    model_path = PROJECT_ROOT / "models" / "demand_model.pkl"
    mapping_path = PROJECT_ROOT / "models" / "family_mapping.pkl"

    assert model_path.exists()
    assert mapping_path.exists()


def test_forecast_returns_dataframe():
    result = forecast_demand(
        1,
        "GROCERY I",
        "2017-08-16",
        7,
        0
    )

    assert isinstance(result, pd.DataFrame)


def test_forecast_has_correct_columns():
    result = forecast_demand(
        1,
        "GROCERY I",
        "2017-08-16",
        7,
        0
    )

    assert "date" in result.columns
    assert "predicted_demand" in result.columns


def test_forecast_has_correct_horizon():
    result = forecast_demand(
        1,
        "GROCERY I",
        "2017-08-16",
        7,
        0
    )

    assert len(result) == 7


def test_predictions_are_numeric():
    result = forecast_demand(
        1,
        "GROCERY I",
        "2017-08-16",
        7,
        0
    )

    assert pd.api.types.is_numeric_dtype(
        result["predicted_demand"]
    )


def test_predictions_are_non_negative():
    result = forecast_demand(
        1,
        "GROCERY I",
        "2017-08-16",
        7,
        0
    )

    assert (result["predicted_demand"] >= 0).all()


def test_forecast_dates_are_sequential():
    result = forecast_demand(
        1,
        "GROCERY I",
        "2017-08-16",
        7,
        0
    )

    dates = pd.to_datetime(result["date"])
    differences = dates.diff().dropna()

    assert (differences == pd.Timedelta(days=1)).all()