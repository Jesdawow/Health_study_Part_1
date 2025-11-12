from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"

HEALTH_FILE = DATA_DIR / "health_study_dataset.csv"

IMAGES_DIR.mkdir(parents=True, exist_ok=True) # Makes sure the image map exists when I want to save images.