# Preprocessing common to all models

# ------ Imports ------ #
import pandas as pd
import numpy as np
from sklearn.utils import resample
import re


# Possible target columns
TARGET_COLS = [
    'IIEF15_01_6m',
    'IIEF15_01_12m',
    'IIEF15_01_24m',
    'IIEF15_01_36m'
]

def create_labels(DATA_PATH_LABELS, TARGET_COLS):
    labels = pd.read_excel(DATA_PATH_LABELS)
    # ------ IIEF15_01 columns ------ #
    iief_cols = ['IIEF15_01_preop'] + TARGET_COLS
    # Round values to nearest integer
    labels[iief_cols] = labels[iief_cols].round()
    # Set values above 5 to NaN
    labels[iief_cols] = labels[iief_cols].where(labels[iief_cols] <= 5)
    return labels

def clean_data(DATA_PATH, DATA_PATH_LABELS):

    data = pd.read_excel(DATA_PATH)
    # ------ lengte ------ # 
    # Remove missing value and values below 100
    data = data[(data['lengte'] > 100) | (data['lengte'].isna())]

    # print(f'Number of patients: {len(data)}')

    # ------ BMI ------ #
    # Impute missing with lengte and gewicht
    data['BMI'] = data['BMI_preop'].fillna(data['gewicht'] / ((data['lengte'] / 100) ** 2))
    # Convert 0 to NaN
    data['BMI'] = data['BMI'].replace(0, np.nan)

    # print(f'Number of patients: {len(data)}')

    # ------ roken_hoeveel ------ #
    # Convert '' to NaN and leave them
    data['roken_hoeveel'] = data['roken_hoeveel'].replace('', np.nan)
    # Remove non-ints
    data['roken_hoeveel'] = data['roken_hoeveel'].replace(r'\D', np.nan, regex=True)
    # Cast to int
    data['roken_hoeveel'] = pd.to_numeric(data['roken_hoeveel'], errors='coerce').astype('Int64')
    # Remove values above 60
    data = data[(data['roken_hoeveel'] < 60) | (data['roken_hoeveel'].isna())]

    # ------ alkohol ------ #
    # Fill NaNs with 1 where alkohol_hoeveel contains a digit
    data['alkohol'] = data['alkohol'].fillna(
        data['alkohol_hoeveel'].apply(
            lambda x: 1 if re.search(r'\d', str(x)) else 0
        )
    )
    # Fill remaining NaNs with 0
    data['alkohol'] = data['alkohol'].fillna(0)

    # print(f'Number of patients: {len(data)}')

    # ------ alkohol_hoeveel ------ #
    # Replace entries containing non-digits with NaN
    data['alkohol_hoeveel'] = data['alkohol_hoeveel'].replace(r'\D', np.nan, regex=True)
    # Convert '' to NaN
    data['alkohol_hoeveel'] = data['alkohol_hoeveel'].replace('', np.nan)
    # Replace NaNs with 0 where alkohol is 0 or 2
    alk_1_or_2 = (data['alkohol'] == 0) | (data['alkohol'] == 2)
    data.loc[alk_1_or_2, 'alkohol_hoeveel'] = data.loc[alk_1_or_2, 'alkohol_hoeveel'].fillna(0)
    # Cast to int
    data['alkohol_hoeveel'] = pd.to_numeric(data['alkohol_hoeveel'], errors='coerce').astype('Int64')
    # Remove values above 60
    data = data[data['alkohol_hoeveel'] < 60]

    labels = create_labels(DATA_PATH_LABELS, TARGET_COLS)
    data = data.merge(labels, on='AnonymizedName', how='left')

    data.rename(columns={
        'IIEF15_01_preop_y': 'IIEF15_01_preop'
    }, inplace=True)

    return data

    # print(f'Number of patients: {len(data)}')



# -------------------------------- FILTERING -------------------------------- #

def filter_data(data, target_col):
    # Use only patients with a 'good' (4-5) preop score
    data = data[data['IIEF15_01_preop'] >= 4]

    # Use only patients with a prime treatement of RALP (Robotic-Assisted Laparoscopic Prostatectomy)
    data = data[data['primtreatment'] == 'RALP']

    # Use only patients with a known target postop score
    data = data.dropna(subset=[f'IIEF15_01_{target_col}'])

    return data

# ------------------------------- UPSAMPLING -------------------------------- #

def upsample_bin_minority(X_train, y_train):
    # Target column
    target_col = y_train.name
    
    # Combine X and y
    X = pd.concat([X_train, y_train], axis=1)

    # Separate majority and minority classes
    class1 = X[X[target_col] == 1]
    class2 = X[X[target_col] == 0]

    # Determine which class is the minority
    minority, majority = (class1, class2) if len(class1) < len(class2) else (class2, class1)

    # Upsample minority class
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

    # Combine majority class with upsampled minority class
    upsampled = pd.concat([majority, minority_upsampled])

    # Shuffle the data
    upsampled = upsampled.sample(frac=1, random_state=42)

    # Return the upsampled data (X, y)
    return upsampled.drop(target_col, axis=1), upsampled[target_col]

# -------------------------------- IMPUTATION ------------------------------- #

def impute_data(data, cat_cols, used_cols):
    # For CAT_COLS, replace NaN with the mode of the column
    for col in cat_cols:
        data = data.fillna({col: data[col].mode()[0]})

    # For the rest of USED_COLS, replace NaN with the median of the column
    for col in used_cols:
        if col not in cat_cols:
            data = data.fillna({col: data[col].median()})

    return data