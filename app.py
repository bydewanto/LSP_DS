import streamlit as st

# Set page config with font settings
st.set_page_config(
    page_title="Home Page",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom HTML to ensure font is applied
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..800;1,400..800&display=swap');
    
    /* Force EB Garamond on all text elements */
    * {
        font-family: 'EB Garamond', serif !important;
    }
    
    .stApp h1, .stApp h2, .stApp h3, .stApp p {
        font-family: 'EB Garamond', serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your content
st.title("Forecast Sales")

st.subheader("Mengenai Dataset yang digunakan")

st.markdown("""---""")

st.write("**Overview**")

st.write("""
    Dataset yang tersedia secara publik ini berisi data historis penjualan dari sebuah toko grosir di Italia, khususnya produk pasta dari empat merek nasional. Dataset ini mencatat data penjualan harian pada tingkat SKU serta informasi promosi selama periode 5 tahun.
""")

st.markdown("""
    - Periode: 1 Januari 2014 - 31 Desember 2018
    - Jumlah Data: 118 deret waktu harian (tingkat SKU)
    - Jumlah Titik Data Deret Waktu: 1.798 baris harian selama 5 tahun
""")

st.markdown("""---""")

st.write("**Dataset Summary Table**")

st.write("""
    | Column Name | Description |
    |-------------|-------------|
    | Rentang Waktu   | Jan 2014 - Des 2018 |
    | Resolusi  | Harian |
    | Produk    | 118 item pasta dari 4 merek |
    | Flag Promosi | Ada (1 untuk iya, 0 untuk tidak) |
    | Missing values | Tidak ada |
""")

st.markdown("---")

st.subheader("**PreProcessing**")

st.markdown("""
    1. Dataset ini dibersihkan dengan menghapus spasi dari nama kolom, mengubah kolom 'DATE' menjadi format datetime, dan menjadikannya sebagai indeks. Hal ini memastikan bahwa data siap untuk analisis deret waktu dengan struktur yang bersih dan konsisten.
    2. Selanjutnya, data dipersiapkan dengan menghasilkan dua fitur utama—jumlah penjualan dan status promosi—berdasarkan mode yang dipilih. Jika mode diatur ke "brand", maka seluruh penjualan SKU dalam brand tersebut dijumlahkan dan promosi akan diberi flag jika ada salah satu SKU yang dipromosikan. Jika mode diatur ke "sku", maka kolom jumlah dan promosi dari item tersebut langsung dipilih dan diubah namanya.
    3. Fitur-fitur baru akan dibuat menggunakan fungsi create_features untuk memperkaya dataset dengan fitur berbasis waktu—seperti hari dalam minggu, bulan, dan tahun—serta statistik lag dan rolling berdasarkan variabel target. Fitur-fitur ini membantu menangkap pola musiman dan tren terkini, sehingga membuat data lebih sesuai untuk model prediksi.
""")

st.markdown("""---""")

st.subheader("**Modeling**")

st.markdown("""
    Proses pemodelan menggunakan model regresi XGBoost untuk memprediksi penjualan di masa depan berdasarkan fitur deret waktu yang telah direkayasa dan data promosi. Data dibagi menjadi set pelatihan dan pengujian, dengan model dilatih untuk mempelajari pola dari data penjualan sebelumnya, termasuk pola musiman dan pengaruh promosi.

Setelah dilatih, model digunakan untuk melakukan multi-step forward forecasting, yaitu memprediksi penjualan harian selama 14 hari ke depan menggunakan nilai terakhir yang diketahui dan memperbarui fitur secara dinamis pada setiap langkah. Akurasi model dievaluasi menggunakan metrik RMSE, dan perbandingan visual antara nilai aktual dan prediksi disediakan untuk menilai performa.
""")

st.page_link("pages/predict.py", label="Go to Predict Page")