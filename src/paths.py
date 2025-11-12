from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" #Use to find the data-map
IMAGES_DIR = ROOT / "images" #Use to find the image-map

HEALTH_FILE = DATA_DIR / "health_study_dataset.csv" #Use to find the dataset file

IMAGES_DIR.mkdir(parents=True, exist_ok=True) # Makes sure the image map exists when I want to save images.