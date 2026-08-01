
import numpy as np
import pandas as pd

from core.constants import BASE_COLS, FEATURE_LABELS_ID


def explain_anomaly_reasons(data, top_k=2, max_rows=200):
    """Untuk tiap nasabah yang ditandai 'Anomaly', cari fitur mana yang paling
    menyimpang dibanding rata-rata nasabah 'Normal', lalu susun jadi kalimat.
    Juga tandai apakah anomalinya dikonfirmasi ganda oleh DBSCAN.

    max_rows membatasi jumlah baris yang diproses/ditampilkan (data anomali
    biasanya ratusan baris - tidak semuanya perlu ditabelkan di dashboard).
    """
    normal = data[data["Anomaly"] == "Normal"]
    means = normal[BASE_COLS].mean()
    stds = normal[BASE_COLS].std().replace(0, np.nan)

    anomalies = data[data["Anomaly"] == "Anomaly"].copy()
    if len(anomalies) > max_rows:
        anomalies = anomalies.sample(max_rows, random_state=42)

    rows = []
    for idx, r in anomalies.iterrows():
        z = ((r[BASE_COLS] - means) / stds).abs().sort_values(ascending=False)
        top_feats = z.head(top_k).index.tolist()

        reasons = []
        for feat in top_feats:
            direction = "jauh di atas" if r[feat] > means[feat] else "jauh di bawah"
            label_col = feat + "_label"
            val_label = r[label_col] if label_col in data.columns else r[feat]
            reasons.append(
                f"{FEATURE_LABELS_ID.get(feat, feat)} ({val_label}) {direction} rata-rata nasabah normal"
            )

        is_dbscan_outlier = r.get("dbscan_cluster", 0) == -1
        jenis = "Anomali ganda (Isolation Forest + DBSCAN)" if is_dbscan_outlier else "Anomali Isolation Forest"

        rows.append({
            "ID Baris": idx,
            "Jenis Anomali": jenis,
            "Faktor Utama": "; ".join(reasons),
            "Status Churn": r.get("Exited_label", r.get("Exited")),
        })

    return pd.DataFrame(rows)


def rules_to_business_insights(rules_df, top_n=10):
    insights = []

    for _, r in rules_df.head(top_n).iterrows():
        ant_feat, ant_val = r["antecedent"].split("=", 1)
        con_feat, con_val = r["consequent"].split("=", 1)

        # 1. Churn
        if con_feat == "Status Churn" and con_val == "Churn":
            sentence = (
                f"Nasabah dengan {ant_feat.lower()} '{ant_val}' memiliki kecenderungan lebih tinggi untuk churn "
                f"(confidence {r['confidence']:.0%}). "
                "Disarankan bank memberikan program retensi seperti penawaran produk tambahan, "
                "promo khusus, atau pendekatan personal sebelum nasabah meninggalkan layanan."
            )

        # 2. Bertahan
        elif con_feat == "Status Churn" and con_val == "Bertahan":
            sentence = (
                f"Nasabah dengan {ant_feat.lower()} '{ant_val}' cenderung tetap bertahan "
                f"(confidence {r['confidence']:.0%}). "
                "Karakteristik ini dapat dijadikan acuan dalam menyusun strategi akuisisi "
                "dan mempertahankan loyalitas nasabah."
            )

        # 3. Produk
        elif con_feat == "Jumlah Produk":
            sentence = (
                f"Nasabah dengan {ant_feat.lower()} '{ant_val}' umumnya menggunakan {con_val.lower()} "
                f"(confidence {r['confidence']:.0%}). "
                "Bank dapat memanfaatkan informasi ini untuk menyusun strategi cross-selling "
                "dan menawarkan produk yang paling relevan."
            )

        # 4. Keaktifan
        elif con_feat == "Status Keaktifan":
            sentence = (
                f"Nasabah dengan {ant_feat.lower()} '{ant_val}' cenderung memiliki status keaktifan "
                f"'{con_val}' (confidence {r['confidence']:.0%}). "
                "Bank dapat meningkatkan engagement melalui program loyalitas dan komunikasi berkala."
            )

        # 5. Negara
        elif con_feat == "Negara":
            sentence = (
                f"Karakteristik '{ant_val}' lebih banyak ditemukan pada nasabah di {con_val}. "
                "Informasi ini dapat dimanfaatkan untuk menyusun strategi pemasaran yang lebih spesifik "
                "pada masing-masing wilayah."
            )

        # 6. Saldo
        elif con_feat == "Saldo":
            sentence = (
                f"Nasabah dengan {ant_feat.lower()} '{ant_val}' umumnya memiliki saldo {con_val.lower()}. "
                "Segmen ini dapat menjadi target penawaran produk investasi atau tabungan sesuai kemampuan finansialnya."
            )

        # Default
        else:
            sentence = (
                f"Nasabah dengan {ant_feat.lower()} '{ant_val}' cenderung memiliki "
                f"{con_feat.lower()} '{con_val}' "
                f"(confidence {r['confidence']:.0%}, lift {r['lift']:.2f}). "
                "Pola ini dapat dijadikan pertimbangan dalam penyusunan strategi bisnis."
            )

        insights.append(sentence)

    return insights


def describe_cluster(sub_df, seg_label):
    """Narasi 1 paragraf yang merangkum karakteristik 1 segmen/klaster."""
    n = len(sub_df)
    if n == 0:
        return f"{seg_label}: tidak ada data."

    churn = sub_df["Exited"].mean()
    geo = sub_df["Geography_label"].mode().iloc[0]
    age = sub_df["Age_label"].mode().iloc[0]
    balance = sub_df["Balance_label"].mode().iloc[0]
    active = sub_df["IsActiveMember_label"].mode().iloc[0]

    return (
        f"{seg_label} berisi {n:,} nasabah ({churn:.1%} di antaranya churn). "
        f"Didominasi nasabah asal {geo}, kelompok usia {age.lower()}, dengan saldo {balance.lower()} "
        f"dan status keaktifan mayoritas '{active.lower()}'."
    )