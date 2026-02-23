import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer

CSV_PATH = "ads_data/globalAdsPerformance.csv"

SEED = 119


TARGET_COLUMN = "revenue"
POSSIBLE_EXCLUDES = [TARGET_COLUMN,"platform","industry","country","date"]

def split_indices(n:int, seed:int, train_frac:float = 0.70, val_frac: float = 0.15):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)

    n_train = int(round(n*train_frac))

    val_train = int(round(n*val_frac))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+val_train]
    test_idx = perm[n_train+val_train:]

    return train_idx,val_idx,test_idx

def add_bias(x):
    new_column = np.ones((x.shape[0],1))
    xb = np.hstack((new_column,x))
    assert xb.shape == (x.shape[0],x.shape[1] + 1)
    return xb
    
def mse_loss(xb,y,w):
    y_hat = xb @ w
    loss = np.mean((y_hat - y)**2)
    return loss

def mse_grad(xb,y,w):
    error = (xb @ w ) - y
    weights = xb.T
    grad = ( (2/len(y)) * (weights @ error))
    return grad


def mse_loss_graph(train_loss,val_loss):
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss Curve – Revenue Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve_rev.png", dpi = 200)
    plt.close()


def regression_plot(y,y_hat,tvt):
    plt.figure()
    plt.scatter(y_hat,y,alpha=0.3)
    plt.ylabel("Actual Revanue")
    plt.xlabel("predicted Revanue")
    plt.title("actual vs predicted y")
    plt.tight_layout
    plt.savefig(f"pred_vs_actual_{tvt}")
    plt.close()

def main():

    df = pd.read_csv(CSV_PATH)
    print(df.head())

    #split the data -> find the numerical and categorial columns --> impute/create pipeline to scale data
    #--> fit data using the training data --> convert to np arrays 
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    x_df = df.drop(columns=POSSIBLE_EXCLUDES)
    test_size = 0.15
    val_size = 0.15 

    x_trainval,x_test,y_trainval,y_test = train_test_split(x_df,y,test_size=test_size,random_state=SEED)

    val_fraction = val_size/(1.0 - test_size)

    x_train,x_val,y_train,y_val = train_test_split(x_trainval,y_trainval,test_size=val_fraction,random_state=SEED)

    print(f"length of train: \n {x_train.shape} \n length of val: \n {x_val.shape} \n length of test: \n {x_test.shape}")
    numeric_col = [c for c in x_df.columns if is_numeric_dtype(x_df[c])]
    cat_col = [c for c in x_df.columns if c not in numeric_col]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scalar",StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num",num_pipe,numeric_col),
            ("cat",cat_pipe,cat_col)
        ],
        remainder="drop"
    )
    x_train_p = pre.fit_transform(x_train)
    x_val_p = pre.transform(x_val)
    x_test_p = pre.transform(x_test)

    x_train_p = np.asarray(x_train_p, dtype=np.float64)
    x_val_p = np.asarray(x_val_p,dtype=np.float64)
    x_test_p = np.asarray(x_test_p,dtype=np.float64)

    xb_tr = add_bias(x_train_p)
    xb_val = add_bias(x_val_p)
    xb_test = add_bias(x_test_p)
    w = np.zeros(xb_tr.shape[1],dtype=np.float64)

    training_losses = []
    val_losses = []
    epochs = 100
    lr = 0.25
    
    for epoch in range(epochs):
        grad = mse_grad(xb_tr,y_train,w)
        w = w - lr * grad
        training_losses.append(mse_loss(xb_tr,y_train,w))
        val_losses.append(mse_loss(xb_val,y_val,w))

        if (epoch+1) % 100 == 0:
            print(epoch+1,training_losses[-1],val_losses[-1])    

    y_hat_train = xb_tr @ w
    y_hat_val = xb_val @ w
    y_hat_test = xb_val @ w

    print(df["revenue"].describe())
    # mse_loss_graph(training_losses,val_losses)
    # regression_plot(y_train,y_hat_train,"train")
    # regression_plot(y_val,y_hat_val,"validate")
    # regression_plot(y_test,y_hat_test,"test")


main()  