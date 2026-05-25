from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2_train = model.score(X_train, y_train)
    r2_test  = model.score(X_test, y_test)

    print(f"Train R²:  {r2_train:.4f}")
    print(f"Test  R²:  {r2_test:.4f}")
    print(f"Test  MAE: {mae:.4f}")
    print(f"Test  RMSE: {rmse:.4f}")
