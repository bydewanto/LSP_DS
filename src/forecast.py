import pandas as pd

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
