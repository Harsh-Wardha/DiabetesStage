"""Shared configuration and immutable project metadata."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Age",
]
BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "DiffWalk",
]
TARGET = "Diabetes_012"
CLASS_MAPPING = {0: "No Diabetes Pattern", 1: "Prediabetes Pattern", 2: "Diabetes Pattern"}
DISCLAIMER = (
    "This application is an educational machine-learning demonstration. It does not "
    "diagnose diabetes, prediabetes or any other medical condition. Only a qualified "
    "healthcare professional using appropriate clinical tests can make a diagnosis."
)

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY") or ("local-development-only" if os.getenv("VERCEL") != "1" else None)
    MAX_CONTENT_LENGTH = 32 * 1024

