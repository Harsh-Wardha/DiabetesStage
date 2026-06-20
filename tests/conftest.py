import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from app import app as flask_app

@pytest.fixture
def app():
    flask_app.config.update(TESTING=True, SECRET_KEY="test")
    return flask_app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def valid_payload():
    return {"HighBP":1,"HighChol":1,"CholCheck":1,"BMI":28.0,"Smoker":0,"Stroke":0,"HeartDiseaseorAttack":0,"PhysActivity":1,"Fruits":1,"Veggies":1,"HvyAlcoholConsump":0,"GenHlth":3,"MentHlth":2,"PhysHlth":2,"DiffWalk":0,"Age":7,"model":"best"}
