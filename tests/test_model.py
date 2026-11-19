import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import predict_rating


def test_prediction_in_valid_range():
    rating = predict_rating("Action", 140, 60.0, 70.0)
    assert 0 <= rating <= 10