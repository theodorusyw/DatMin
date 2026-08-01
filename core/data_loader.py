import pandas as pd
from .constants import MAPS

DATA_PATH = "data/after_phase4.csv"
df_raw = pd.read_csv(DATA_PATH)

df = df_raw.copy()
for col, mapping in MAPS.items():
    df[col + "_label"] = df[col].map(mapping)
df["Tenure_label"] = df["Tenure"].astype(str)
df["NumOfProducts_label"] = df["NumOfProducts"].astype(str)

N_TOTAL = len(df)
CHURN_RATE = df["Exited"].mean()
ANOMALY_COUNT = int((df["Anomaly"] == "Anomaly").sum())
ANOMALY_RATE = ANOMALY_COUNT / N_TOTAL
OUTLIER_DBSCAN_COUNT = int((df["dbscan_cluster"] == -1).sum())
CHURN_RATE_DBSCAN_OUTLIER = df.loc[df["dbscan_cluster"] == -1, "Exited"].mean()
CHURN_RATE_DBSCAN_NORMAL = df.loc[df["dbscan_cluster"] == 0, "Exited"].mean()
CHURN_RATE_ANOMALY = df.loc[df["Anomaly"] == "Anomaly", "Exited"].mean()
CHURN_RATE_NORMAL = df.loc[df["Anomaly"] == "Normal", "Exited"].mean()