import pytest
import pandas as pd
from ..models.models import preprocessing, model_catboost
import sys
import os

def test_preprocessing():
    try:
        df = preprocessing()
        assert 'Attrition' in df.columns
        assert df['Attrition'].dtype in [int, float]
        assert 'EmployeeNumber' not in df.columns
    except FileNotFoundError:
        pytest.skip("Файл данных не найден")

def test_model_training():
    result = model_catboost()
    assert 'model' in result
    assert 'recall' in result
    assert result['recall'] > 0.7
    assert 'X_test' in result