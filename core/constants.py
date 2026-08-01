BASE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

MAPS = {
    "Geography": {0: "Prancis", 1: "Jerman", 2: "Spanyol"},
    "Gender": {0: "Wanita", 1: "Pria"},
    "CreditScore": {0: "Rendah (350-580)", 1: "Sedang (580-670)",
                    2: "Tinggi (670-740)", 3: "Sangat Tinggi (740-850)"},
    "Age": {0: "Muda (18-30 th)", 1: "Dewasa (31-45 th)", 2: "Senior (46+ th)"},
    "Balance": {0: "Rendah (\u2264 50rb)", 1: "Sedang (50-100rb)",
                2: "Tinggi (100-150rb)", 3: "Sangat Tinggi (>150rb)"},
    "EstimatedSalary": {0: "Rendah (\u2264 50rb)", 1: "Menengah (50-100rb)",
                        2: "Tinggi (100-150rb)", 3: "Sangat Tinggi (>150rb)"},
    "HasCrCard": {0: "Tidak Punya", 1: "Punya"},
    "IsActiveMember": {0: "Tidak Aktif", 1: "Aktif"},
    "Exited": {0: "Bertahan", 1: "Churn"},
}

FEATURE_LABELS_ID = {
    "CreditScore": "Skor Kredit",
    "Geography": "Negara",
    "Gender": "Gender",
    "Age": "Usia",
    "Tenure": "Lama Menjadi Nasabah (tahun)",
    "Balance": "Saldo",
    "NumOfProducts": "Jumlah Produk",
    "HasCrCard": "Kartu Kredit",
    "IsActiveMember": "Status Keaktifan",
    "EstimatedSalary": "Estimasi Gaji",
    "Exited": "Status Churn",
}

CLUSTER_METHODS = {
    "cluster": "K-Modes (3 Segmen)",
    "cluster_kmodes": "K-Modes (2 Segmen)",
    "cluster_kmeans": "K-Means (2 Segmen)",
    "dbscan_cluster": "DBSCAN (Deteksi Kelompok Padat vs Outlier)",
}

FEATURE_COLOR_PALETTE = {
    "Skor Kredit": "#2EC4B6", "Negara": "#1B4B66", "Gender": "#8E7CC3",
    "Usia": "#E8871E", "Lama Menjadi Nasabah (tahun)": "#5B8C5A",
    "Saldo": "#3D5A80", "Jumlah Produk": "#C2A33E", "Kartu Kredit": "#7A9E9F",
    "Status Keaktifan": "#E63946", "Estimasi Gaji": "#6A4C93",
    "Status Churn": "#D62839",
}

COLORS = {
    "navy": "#0F2A43", "navy2": "#1B4B66", "teal": "#2EC4B6",
    "amber": "#E8871E", "danger": "#D62839", "bg": "#F5F7FA",
    "muted": "#6B7C93",
}