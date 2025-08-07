import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from src.features import create_features
import matplotlib.pyplot as plt

def train_xgboost(df, target_col, promo_col):
    df = create_features(df, target_col, promo_col)
    FEATURES = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear',
                'dayofmonth', 'weekofyear', 'lag_1', 'lag_7', 'rolling_mean_7',
                'rolling_std_7', promo_col]

    train = df[df.index < '2018-01-06']
    test = df[df.index >= '2018-01-06']

    X_train = train[FEATURES]
    y_train = train[target_col]
    X_test = test[FEATURES]
    y_test = test[target_col]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        early_stopping_rounds=50,
        learning_rate=0.01,
        max_depth=3,
        objective='reg:squarederror'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )

    test['prediction'] = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, test['prediction']))

    test[[target_col, 'prediction']].plot(figsize=(15, 5), title=f"Prediksi vs Aktual - {target_col}")
    plt.ylabel("Jumlah Penjualan")
    plt.xlabel("Tanggal")
    plt.show()

    return model, test, rmse
