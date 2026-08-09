import pytest
import numpy as np
import pandas as pd
from backend.model import RegdateMonthsEncoder, CategoricalInteractionEncoder, LinearRegressionModel

@pytest.fixture
def regdate_months_encoder():
    return RegdateMonthsEncoder()

@pytest.fixture
def categorical_interaction_encoder():
    return CategoricalInteractionEncoder(drop="first", handle_unknown="ignore")

@pytest.fixture
def month_data():
    return np.array([['2026-01-01'], ['2026-02-01']])

@pytest.fixture
def month_data_bad():
    return np.array([['hello'], ['world']])

@pytest.fixture
def data_2row_2col_random():
    rng = np.random.default_rng(seed=42)
    return rng.random(size=(3, 4))

@pytest.fixture
def data_full_small():
    return pd.DataFrame({
        'date': ['2026-01-01', '2026-02-01', '2025-08-01', '2025-10-01'],
        'town': ["Bishan", "Bishan", "Yishun", "Yishun"],
        'flat_type': ["4 ROOM", "4 ROOM", "4 ROOM", "3 ROOM"],
        'floor_area_sqm': [121, 82, 84, 67],
        'resale_price': [1234, 5678, 2000, 4000],
    })

@pytest.fixture
def data_full_more_unique():
    return pd.DataFrame({
        'date': ['2026-01-01', '2026-02-01', '2025-08-01', '2025-10-01'],
        'town': ["Bishan", "Toa Payoh", "Yishun", "Queenstown"],
        'flat_type': ["5 ROOM", "4 ROOM", "4 ROOM", "3 ROOM"],
        'floor_area_sqm': [121, 82, 84, 67],
        'resale_price': [1234, 5678, 2000, 4000],
    })

def test_RegdateMonthsEncoder_shape(regdate_months_encoder, month_data):
    res = regdate_months_encoder.fit_transform(month_data)
    assert res.shape == (2, 1)

def test_RegdateMonthsEncoder_correct_output(regdate_months_encoder, month_data):
    res = regdate_months_encoder.fit_transform(month_data)
    assert res[0, 0] == 0
    assert res[1, 0] == 1

def test_RegdateMonthsEncoder_bad_input(regdate_months_encoder, month_data_bad):
    with pytest.raises(ValueError):
        res = regdate_months_encoder.fit_transform(month_data_bad)

def test_RegdateMonthsEncoder_bad_transform(regdate_months_encoder, month_data, data_2row_2col_random):
    transformer = regdate_months_encoder.fit(month_data)
    with pytest.raises(ValueError):
        res = transformer.transform(data_2row_2col_random)

def test_CategoricalInteractionEncoder(categorical_interaction_encoder, data_full_more_unique):
    # For drop="first", OHE always drops the first value in ascending order
    # So "Bishan" and "3 ROOM" columns should be dropped. The corresponding interactions are also dropped in the one-hot encoding.
    # This leaves us with 3 (town) + 2 (flat type) + 2 (interactions) = 7 columns
    # Number of rows should be preserved.

    res = categorical_interaction_encoder.fit_transform(data_full_more_unique[["town", "flat_type"]])
    assert res.shape == (4, 7)

def test_LinearRegressionModel_feature_eng_small(data_full_small):
    # "Bishan" and "3 ROOM" should be "dropped", leaving us with 5 columns (date, floor_area, 3 ohe cols for town, flat_type, town_flat_type)
    train = data_full_small[['date', 'town', 'flat_type', 'floor_area_sqm']]
    y = data_full_small['resale_price']
    model = LinearRegressionModel().fit(train, y)
    res = model.get_features(data_full_small)
    assert res.shape == (4, 5)

def test_LinearRegressionModel_feature_eng_more_unique(data_full_more_unique):
    # "Bishan" and "3 ROOM" should be "dropped", leaving us with 9 columns (date, floor_area, town (3 cols), flat_type (2 cols), town_flat_type (2 cols))
    train = data_full_more_unique[['date', 'town', 'flat_type', 'floor_area_sqm']]
    y = data_full_more_unique['resale_price']
    model = LinearRegressionModel().fit(data_full_more_unique, y)
    res = model.get_features(data_full_more_unique)
    assert res.shape == (4, 9)

def test_LinearRegressionModel_feature_eng_train_test(data_full_more_unique, data_full_small):
    # In this test we fit the training set (which gives 9 feature columns) and 
    # see if the model transforms the test set into a feature set of the same number of columns.

    train = data_full_more_unique[['date', 'town', 'flat_type', 'floor_area_sqm']]
    y = data_full_more_unique['resale_price']
    test = data_full_small[['date', 'town', 'flat_type', 'floor_area_sqm']]

    model = LinearRegressionModel().fit(train, y)
    res = model.get_features(test)
    assert res.shape == (4, 9)


