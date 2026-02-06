import numpy as np # type: ignore
import pandas as pd # type: ignore

CSV_PATH = "ads_data/globalAdsPerformance.csv"

SEED = 119


TARGET_COLUMN = "revenue"
POSSIBLE_EXCLUDES = ["pid",TARGET_COLUMN,"platform","industry","country","date"]

def split_indices(n:int, seed:int, train_frac:float = 0.70, val_frac: float = 0.15):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)

    n_train = int(round(n*train_frac))

    val_train = int(round(n*val_frac))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+val_train]
    test_idx = perm[n_train+val_train:]

    return train_idx,val_idx,test_idx

def main():

    df = pd.read_csv(CSV_PATH)

    train_idx,val_idx,test_idx = split_indices(len(df),SEED)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # get the column we want to predict

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype= float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype = float)

    x_train = train_df.drop(columns = [c for c in POSSIBLE_EXCLUDES if c in train_df.columns])
    x_val = train_df.drop(columns = [c for c in POSSIBLE_EXCLUDES if c in val_df])
    x_test = train_df.drop(columns= [c for c in POSSIBLE_EXCLUDES if c in test_df])

    numeric_col = [c for c in x_train.columns if pd.api.types.is_numeric_dtype(x_train[c]) ]
    cat_col = [c for c in x_train.columns if c not in numeric_col]

    x_train_num = x_train[numeric_col]
    x_val_num = x_val[numeric_col]
    x_test_num = x_test[numeric_col]

    x_train_cat = x_train[cat_col]
    x_val_cat = x_train[cat_col]
    x_test_cat = x_train[cat_col]

    x_train_median = x_train_num.median()

    x_train_mode = x_train_cat.mode()

    print(x_train_num.isna().any())

    x_train_num_imputed = x_train_num.fillna(x_train_median)
    x_val_num_imputed = x_val_num.fillna(x_train_median)
    x_test_num_imputed = x_test_num.fillna(x_train_median)


    # for index,value in x_train_num_nas.items():
    #     if value:
    #         x_train_num_imputed = x_train_num.fillna(x_train_median)

    x_train_cat_imputed = x_train_cat.fillna(x_train_mode.iloc[0])
    x_val_cat_imputed = x_val_cat.fillna(x_train_mode.iloc[0])
    x_test_cat_imputed = x_val.fillna(x_train_mode.iloc[0])

    x_train_mean = x_train_num_imputed.mean()
    x_train_std = x_train_num_imputed.std()

    scaled_x_train = ((x_train_num_imputed-x_train_mean)/x_train_std)
    scaled_x_val = ((x_val_num_imputed-x_train_mean)/x_train_std)
    scaled_x_test = ((x_train_num_imputed-x_train_mean)/x_train_std)

    x_train_ho = pd.get_dummies(x_train_cat_imputed)
    x_train_ho_columns = x_train_ho.columns
    x_val_ho = pd.get_dummies(x_val_cat_imputed).reindex(columns= x_train_ho_columns,fill_value=0)
    x_test_ho = pd.get_dummies(x_test_cat_imputed).reindex(columns= x_train_ho_columns,fill_value=0)

    x_train_np = np.concatenate((scaled_x_train,x_train_ho),axis=1)
    x_val_np = np.concatenate((scaled_x_val,x_val_ho), axis=1)
    x_test_np = np.concatenate((scaled_x_test,x_test_ho), axis =1)
            
            
    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
    print("Numeric cols:", numeric_col)
    print("Categorical cols:", cat_col)

    print(x_train_np.shape,y_train.shape)
main()  