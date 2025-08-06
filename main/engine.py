# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('hierarchical_data.csv')

# Preprocessing dasar
df.columns = df.columns.str.strip()
print(df.columns) # Print columns to check
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')

# Fungsi modular untuk SKU dan brand

def prepare_data(df, mode="brand", id="B1"):
    df = df.copy()
    if mode == "brand":
        qty_cols = [col for col in df.columns if col.startswith(f"QTY_{id}_")]
        promo_cols = [col for col in df.columns if col.startswith(f"PROMO_{id}_")]
        df[f"QTY_{mode.upper()}_{id}"] = df[qty_cols].sum(axis=1)
        df[f"PROMO_{mode.upper()}_{id}"] = df[promo_cols].max(axis=1)
        result = df[[f"QTY_{mode.upper()}_{id}", f"PROMO_{mode.upper()}_{id}"]].copy()
    elif mode == "sku":
        result = df[[f"QTY_{id}", f"PROMO_{id}"].copy()]
        result = result.rename(columns={f"QTY_{id}": f"QTY_{mode.upper()}_{id}", f"PROMO_{id}": f"PROMO_{mode.upper()}_{id}"})
    result.index = df.index
    return result

# Buat fitur waktu dan lag

def create_features(df, target_col, promo_col):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    # Lag features
    df['lag_1'] = df[target_col].shift(1)
    df['lag_7'] = df[target_col].shift(7)
    df['rolling_mean_7'] = df[target_col].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df[target_col].shift(1).rolling(window=7).std()

    df = df.dropna()
    return df

# Training dan evaluasi

def train_xgboost(df, target_col, promo_col):
    df = create_features(df, target_col, promo_col)
    FEATURES = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', promo_col]

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
        objective='reg:squarederror',
        verbosity=1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )

    test['prediction'] = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, test['prediction']))
    print(f"\n\U0001F4A1 RMSE pada test set: {rmse:.2f}")

    # Plot prediksi vs aktual
    test[[target_col, 'prediction']].plot(figsize=(15, 5), title=f"Prediksi vs Aktual - {target_col}")
    plt.ylabel("Jumlah Penjualan")
    plt.xlabel("Tanggal")
    plt.show()

    return model, test, rmse

# Fungsi multistep forward forecasting

def forecast_next_days(model, last_known_df, n_steps, target_col, promo_col, promo_value=0):
    forecast = []
    current = last_known_df.copy()

    for i in range(n_steps):
        next_index = current.index[-1] + pd.Timedelta(days=1)
        next_row = {
            'dayofweek': next_index.dayofweek,
            'quarter': next_index.quarter,
            'month': next_index.month,
            'year': next_index.year,
            'dayofyear': next_index.dayofyear,
            'dayofmonth': next_index.day,
            'weekofyear': next_index.isocalendar().week,
            'lag_1': current[target_col].iloc[-1],
            'lag_7': current[target_col].iloc[-7] if len(current) >= 7 else current[target_col].iloc[-1],
            'rolling_mean_7': current[target_col].rolling(window=7).mean().iloc[-1] if len(current) >= 7 else current[target_col].mean(),
            'rolling_std_7': current[target_col].rolling(window=7).std().iloc[-1] if len(current) >= 7 else current[target_col].std(),
            promo_col: promo_value
        }
        X_next = pd.DataFrame([next_row], index=[next_index])
        y_pred = model.predict(X_next)[0]
        current.loc[next_index] = {**next_row, target_col: y_pred}
        forecast.append((next_index, y_pred))

    forecast_df = pd.DataFrame(forecast, columns=["date", "forecast"])
    forecast_df.set_index("date", inplace=True)
    return forecast_df

# Contoh penggunaan: Brand B1
mode = "brand"
id = "B1"
df_prepared = prepare_data(df, mode=mode, id=id)
model, test_result, rmse = train_xgboost(df_prepared, target_col=f"QTY_{mode.upper()}_{id}", promo_col=f"PROMO_{mode.upper()}_{id}")

# 9. Forecast 14 hari ke depan dari data terakhir
df_features = create_features(df_prepared, target_col=f"QTY_{mode.upper()}_{id}", promo_col=f"PROMO_{mode.upper()}_{id}")
forecast_14d = forecast_next_days(model, df_features, n_steps=14, target_col=f"QTY_{mode.upper()}_{id}", promo_col=f"PROMO_{mode.upper()}_{id}")

print("\nForecast 14 Hari Ke Depan:")
print(forecast_14d)

forecast_14d.plot(figsize=(12,5), title="Prediksi 14 Hari ke Depan")
plt.ylabel("Jumlah Penjualan")
plt.xlabel("Tanggal")
plt.show()