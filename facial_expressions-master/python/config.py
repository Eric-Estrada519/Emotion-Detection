import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "..", "images")

DATA_DIR = os.path.join(BASE_DIR, "..", "data")

LEGEND_CSV = os.path.join(DATA_DIR, "legend.csv")
TRAIN_CSV  = os.path.join(DATA_DIR, "train.csv")
VAL_CSV    = os.path.join(DATA_DIR, "val.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test.csv")
