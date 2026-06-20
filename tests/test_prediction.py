import pytest
from services.prediction_service import predict
from services.validation_service import ValidationError

def test_valid_prediction(valid_payload):
    result=predict(valid_payload)
    assert result["class_id"] in [0,1,2]
    assert 0 <= result["confidence"] <= 100
    assert abs(sum(result["probabilities"].values())-100) < .2

def test_api(client, valid_payload):
    response=client.post("/api/predict", json=valid_payload)
    assert response.status_code == 200 and len(response.json["probabilities"]) == 3

@pytest.mark.parametrize("field,value", [("BMI",9),("BMI",101),("Age",14),("HighBP",2)])
def test_invalid_ranges(valid_payload, field, value):
    valid_payload[field]=value
    with pytest.raises(ValidationError): predict(valid_payload)

def test_missing(valid_payload):
    valid_payload.pop("BMI")
    with pytest.raises(ValidationError): predict(valid_payload)

def test_unknown_model(valid_payload):
    valid_payload["model"]="unknown"
    with pytest.raises(ValidationError): predict(valid_payload)
