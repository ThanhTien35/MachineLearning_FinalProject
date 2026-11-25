from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(seed=42):
    X, y = load_diabetes(return_X_y=True, scaled=False)
    y = y.reshape(-1, 1)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )
    # Chi fit scaler tren tap huan luyen de tranh ro ri du lieu.
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    train = (x_scaler.transform(X_train), y_scaler.transform(y_train))
    val = (x_scaler.transform(X_val), y_scaler.transform(y_val))
    test = (x_scaler.transform(X_test), y_test)
    return train, val, test, y_scaler
