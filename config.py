DEFAULT_RANKING=150
CONFIDENCE_DATA_PATH="Generated Data/confidence_data.csv"
DATA_FILES_PATH="Data/20*.xls*"
ATP_DATA_PATH="Generated Data/atp_data.csv"
ATP_DATA_FEATURES_PATH="Generated Data/atp_data_features.csv"
REGENERATE_FEATURES=True

# XGB parameters
XGB_LEARNING_RATE=[0.1]
XGB_MAX_DEPTH=[5]
XGB_MIN_CHILD_WEIGHT=[1]
XGB_GAMMA=[0.25]
XGB_CSBT=[0.5]
XGB_LAMBDA=[0]
XGB_ALPHA=[2]
XGB_NUM_ROUNDS=[300]
XGB_EARLY_STOP=[5]
XGB_SUBSAMPLE=[0.8]

# Function to get XGB params as numpy array
def get_xgb_params():
    import numpy as np
    params = np.array(np.meshgrid(
        XGB_LEARNING_RATE,
        XGB_MAX_DEPTH,
        XGB_MIN_CHILD_WEIGHT,
        XGB_GAMMA,
        XGB_CSBT,
        XGB_LAMBDA,
        XGB_ALPHA,
        XGB_NUM_ROUNDS,
        XGB_EARLY_STOP,
        XGB_SUBSAMPLE
    )).T.reshape(-1, 10).astype(np.float64)
    return params[0]