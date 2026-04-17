from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_ACCIDENTS_DIR = PROJECT_ROOT / "data" / "raw_accidents"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

YEARS = [2012, 2013, 2014, 2015, 2016]

SELECTED_AREA_NAME = "leeds_central_selected_box"

CENTER_LAT = 53.79997
CENTER_LON = -1.54089

BBOX_WGS84 = {
    "north": 53.80488,
    "south": 53.79505,
    "east": -1.53249,
    "west": -1.54929,
}

BBOX_BNG = {
    "west": 429786.85,
    "east": 430886.85,
    "south": 433287.94,
    "north": 434387.94,
}