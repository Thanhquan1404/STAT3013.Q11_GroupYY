# config.py

# ---- Data config ----
CSV_PATH = r"D:\PTTK\normalized_cirrhosis.csv"
TEST_CSV_PATH = r"D:\PTTK\normalized_cirrhosis.csv" # Test data tuong trung

TARGET_COL = "Stage"   # <---- NOW USING MULTICLASS LABEL (1,2,3)

FEATURE_COLS = [
    # "Status",
    # "Age",
    "Drug",
    "Sex",
    "Ascites",
    "Hepatomegaly",
    "Spiders",
    "Edema",
    "Bilirubin",
    "Cholesterol",
    "Albumin",
    "Copper",
    "Alk_Phos",
    "SGOT",
    "Tryglicerides",
    "Platelets",
    "Prothrombin"
]

RANDOM_STATE = 42
N_SPLITS = 6

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced"     # works fine in multiclass
}

MODEL_PATH = r"D:\VS_Code\DataScience\random_forest_pttk\models\rf_stage.pkl"
ENSEMBLE_MODEL_PATTERN = r"D:\VS_Code\DataScience\random_forest_pttk\models\rf_stage_fold_{fold}.pkl"