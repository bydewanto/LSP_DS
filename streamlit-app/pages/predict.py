import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.model import train_xgboost
from src.features import create_features
from src.forecast import forecast_next_days

# --- Font Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..800;1,400..800&display=swap');
    .stApp h1, .stApp h2, .stApp h3, .stApp p {
        font-family: 'EB Garamond', serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Title and Instructions ---
st.title("📈 Forecast Penjualan")
st.write("Masukkan parameter di bawah untuk memproses prediksi:")

# --- Sidebar Inputs ---
mode = st.radio("Pilih Jenis Prediksi", ["Brand", "SKU"])
mode = mode.lower()

if mode == "brand":
    id = st.selectbox("Pilih Brand ID", ["B1", "B2", "B3", "B4"])
else:
    id = st.selectbox("Pilih SKU ID", [
        "B1_1", "B1_2", "B1_3",
        "B2_1", "B2_2", "B2_3",
        "B3_1", "B3_2", "B3_3",
        "B4_1", "B4_2", "B4_3"
    ])

promo_value = st.radio("Apakah ada promo saat prediksi?", ["Ya", "Tidak"])
n_days = st.selectbox("Jumlah Hari Prediksi", [7, 14, 30])

# --- Run Forecast Button ---
if st.button("🔮 Proses Prediksi"):
    with st.spinner("Sedang memproses model dan prediksi..."):
        promo_input = 1 if promo_value == "Ya" else 0

        # Load and prepare data
        df = load_data("hierarchical_sales_data.csv")
        df_prepared = prepare_data(df, mode=mode, id=id)

        target_col = f"QTY_{mode.upper()}_{id}"
        promo_col = f"PROMO_{mode.upper()}_{id}"

        # Train model
        model, test_result, rmse = train_xgboost(df_prepared, target_col, promo_col)

        # Create features and forecast
        df_features = create_features(df_prepared, target_col, promo_col)
        forecast = forecast_next_days(
            model=model,
            last_known_df=df_features,
            n_steps=n_days,
            target_col=target_col,
            promo_col=promo_col,
            promo_value=promo_input
        )

    # --- Output Display ---
    st.subheader("📉 Hasil RMSE Model")
    st.success(f"RMSE: {rmse:.2f}")

    # Use expander instead of button
    with st.expander("Apa itu RMSE?"):
        st.info(
            "RMSE (Root Mean Square Error) adalah ukuran seberapa jauh prediksi model dari nilai sebenarnya. "
            "Semakin rendah nilai RMSE, semakin baik performa model dalam memprediksi."
        )

    st.subheader(f"📊 Forecast {n_days} Hari ke Depan")
    st.dataframe(forecast)

    # --- Forecast Plot ---
    fig, ax = plt.subplots(figsize=(12, 5))
    forecast.plot(ax=ax, title=f"Prediksi {n_days} Hari ke Depan untuk {mode.upper()} {id}")
    plt.ylabel("Jumlah Penjualan")
    plt.xlabel("Tanggal")
    st.pyplot(fig)
