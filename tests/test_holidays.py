import pytest
import pandas as pd
from mlops_exo.features.holidays import HolidaysComputer

@pytest.mark.parametrize(
    "date_str, expected_year, expected_month, expected_week, expected_day", [
        ("2025-01-01", 2025, 1, 1, 1),
        ("2025-12-25", 2025, 12, 52, 25),
    ]
)
def test_extract_dates(date_str, expected_year, expected_month, expected_week, expected_day):
    # Given
    holidays_computer = HolidaysComputer()
    data = {'Date': [date_str]}
    df = pd.DataFrame(data)

    #When
    result = holidays_computer.extract_dates(df)

    #Then
    assert result['Year'][0] == expected_year
    assert result['Month'][0] == expected_month
    assert result['Week'][0] == expected_week
    assert result['Day'][0] == expected_day