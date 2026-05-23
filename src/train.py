from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import preprocess

def train_model(df=None):

    if df is None:
        df = preprocess.preprocess_data()

    train_df = df[df["year"] < 2025].drop(columns="year")
    test_df  = df[df["year"] == 2025].drop(columns="year")

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Expected non-empty train and test splits, "
            f"got {len(train_df)} train rows and {len(test_df)} test rows."
        )

    X_train = train_df.drop(columns="demand").values
    y_train = train_df["demand"].values
    X_test  = test_df.drop(columns="demand").values
    y_test  = test_df["demand"].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2_train = model.score(X_train, y_train)
    r2_test  = model.score(X_test, y_test)

    print(f"Train R²:  {r2_train:.4f}")
    print(f"Test  R²:  {r2_test:.4f}")
    print(f"Test  MAE: {mae:.4f}")
    print(f"Test  RMSE: {rmse:.4f}")

    return model


