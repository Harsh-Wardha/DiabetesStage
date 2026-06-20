import pandas as pd
import pytest
from scripts.prepare_dataset import validate_three_classes

def test_binary_target_rejected():
    with pytest.raises(ValueError, match="Binary target"):
        validate_three_classes(pd.Series([0,1,0]))

def test_three_classes_accepted():
    assert validate_three_classes(pd.Series([0,1,2])) == [0,1,2]
