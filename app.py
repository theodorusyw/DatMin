import numpy as np
import pandas as pd

import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from core.constants import *
from core.data_loader import *
from core.pca import PCA_VAR_EXPLAINED
from core.mining import (
    mine_association_rules,
    top_rule_involving,
    _default_rules
)
from core.insights import explain_anomaly_reasons, rules_to_business_insights, describe_cluster

from layouts.components import kpi_card, build_navbar

def label_clusters_by_churn(data, col):
    """Beri label bisnis (Risiko Tinggi/Rendah) berdasarkan churn rate tiap cluster."""
    if col == "dbscan_cluster":
        return {-1: "Outlier (di luar pola umum)", 0: "Klaster Utama (pola umum)"}
    rates = data.groupby(col)["Exited"].mean().sort_values(ascending=False)
    n = len(rates)
    labels = {}
    for rank, (cluster_val, rate) in enumerate(rates.items()):
        if n <= 2:
            tag = "Segmen Risiko Lebih Tinggi" if rank == 0 else "Segmen Risiko Lebih Rendah"
        else:
            if rank == 0:
                tag = "Segmen Risiko Tinggi"
            elif rank == n - 1:
                tag = "Segmen Loyal (Risiko Rendah)"
            else:
                tag = "Segmen Risiko Menengah"
        labels[cluster_val] = f"{tag} (Klaster {cluster_val})"
    return labels



def _feature_of(item_name):
    return item_name.split("=")[0]


def build_rule_network_figure(rules_df, top_n=20):
    if rules_df.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="Tidak ada aturan yang memenuhi ambang batas ini.\nCoba turunkan nilai minimum support/confidence.",
                               showarrow=False, font=dict(size=14))],
            template="plotly_white", height=520,
        )
        return fig

    rules_top = rules_df.head(top_n).reset_index(drop=True)
    items = pd.unique(rules_top[["antecedent", "consequent"]].values.ravel())
    n = len(items)
    angles = {item: 2 * np.pi * i / n for i, item in enumerate(items)}
    pos = {item: (np.cos(angles[item]), np.sin(angles[item])) for item in items}

    max_lift = rules_top["lift"].max()
    min_lift = rules_top["lift"].min()

    annotations = []
    for _, r in rules_top.iterrows():
        x0, y0 = pos[r["antecedent"]]
        x1, y1 = pos[r["consequent"]]
        lift_norm = 0.5 if max_lift == min_lift else (r["lift"] - min_lift) / (max_lift - min_lift)
        opacity = 0.25 + 0.55 * lift_norm
        annotations.append(dict(
            ax=x0, ay=y0, x=x1, y=y1, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.1,
            arrowwidth=1 + r["confidence"] * 3,
            arrowcolor=f"rgba(27,75,102,{opacity:.2f})",
            standoff=16, startstandoff=16,
        ))

    node_colors = [FEATURE_COLOR_PALETTE.get(_feature_of(it), "#999999") for it in items]
    node_trace = go.Scatter(
        x=[pos[it][0] for it in items], y=[pos[it][1] for it in items],
        mode="markers+text",
        text=[it.split("=")[1] if "=" in it else it for it in items],
        textposition="top center",
        textfont=dict(size=11, color="#1C2B36"),
        marker=dict(size=22, color=node_colors, line=dict(width=2, color="white")),
        hovertext=items, hoverinfo="text", showlegend=False,
    )

    legend_traces = []
    seen_features = []
    for it in items:
        feat = _feature_of(it)
        if feat not in seen_features:
            seen_features.append(feat)
            legend_traces.append(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=FEATURE_COLOR_PALETTE.get(feat, "#999999")),
                name=feat, showlegend=True,
            ))

    fig = go.Figure(data=[node_trace] + legend_traces)
    fig.update_layout(
        annotations=annotations,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
        xaxis=dict(visible=False, range=[-1.4, 1.4]),
        yaxis=dict(visible=False, range=[-1.4, 1.4], scaleanchor="x", scaleratio=1),
        template="plotly_white",
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Bank Customer Insight Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server


#RINGKASAN EKSEKUTIF

def render_tab_overview():
    churn_by_geo = df.groupby("Geography_label")["Exited"].mean().reset_index()
    fig_geo = px.bar(
        churn_by_geo, x="Geography_label", y="Exited",
        text=churn_by_geo["Exited"].apply(lambda v: f"{v:.1%}"),
        color="Exited", color_continuous_scale=["#2EC4B6", "#D62839"],
        labels={"Geography_label": "Negara", "Exited": "Tingkat Churn"},
    )
    fig_geo.update_traces(textposition="outside")
    fig_geo.update_layout(template="plotly_white", showlegend=False, coloraxis_showscale=False,
                           yaxis_tickformat=".0%", height=340, margin=dict(t=20, b=20, l=20, r=20))

    fig_donut = go.Figure(data=[go.Pie(
        labels=["Bertahan", "Churn"],
        values=[N_TOTAL - int(df["Exited"].sum()), int(df["Exited"].sum())],
        hole=0.6, marker=dict(colors=[COLORS["teal"], COLORS["danger"]]),
    )])
    fig_donut.update_layout(template="plotly_white", height=340, margin=dict(t=20, b=20, l=20, r=20),
                             annotations=[dict(text=f"{CHURN_RATE:.1%}<br>Churn", showarrow=False, font_size=16)])

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Total Nasabah", f"{N_TOTAL:,}", "data historis dianalisis"), md=3),
            dbc.Col(kpi_card("Tingkat Churn", f"{CHURN_RATE:.1%}", "dari seluruh nasabah", COLORS["danger"]), md=3),
            dbc.Col(kpi_card("Segmen Teridentifikasi", "3 Segmen Utama", "hasil K-Modes clustering", COLORS["teal"]), md=3),
            dbc.Col(kpi_card("Anomali Terdeteksi", f"{ANOMALY_COUNT:,} ({ANOMALY_RATE:.1%})", "via Isolation Forest", COLORS["amber"]), md=3),
        ], className="g-3 mb-4"),

        dbc.Card(dbc.CardBody([
            html.H5("Pertanyaan Sentral: Apa yang kita temukan yang tidak terlihat jelas dari data mentah?", className="mb-3"),
            html.Ul([
                html.Li([html.B("Bukan jumlah produk sedikit yang aman, justru sebaliknya. "),
                         f"Nasabah yang churn memiliki kecenderungan kuat hanya memakai 1 produk "
                         f"(pola ini muncul di {_default_rules[_default_rules['consequent'].str.contains('Jumlah Produk', na=False)]['confidence'].max():.0%} kasus jika ditemukan, lihat tab Pattern Mining untuk detail) sinyal ini tidak terlihat hanya dari melihat rata-rata jumlah produk secara keseluruhan."]),
                html.Li([html.B("Nasabah 'tidak biasa' secara statistik jauh lebih berisiko. "),
                         f"Nasabah yang terdeteksi sebagai outlier oleh DBSCAN memiliki tingkat churn "
                         f"{CHURN_RATE_DBSCAN_OUTLIER:.1%}, jauh di atas nasabah pada pola umum ({CHURN_RATE_DBSCAN_NORMAL:.1%}). "
                         "Pola ini baru terlihat setelah clustering tidak tampak dari tabel data mentah."]),
                html.Li([html.B("Isolation Forest mengonfirmasi temuan yang sama dari sudut berbeda. "),
                         f"Nasabah yang ditandai 'Anomaly' punya tingkat churn {CHURN_RATE_ANOMALY:.1%}, "
                         f"lebih dari 2x lipat dibanding nasabah 'Normal' ({CHURN_RATE_NORMAL:.1%})."]),
                html.Li([html.B("Segmentasi mengungkap kelompok nasabah dengan risiko sangat berbeda "),
                         "meskipun rata-rata keseluruhan terlihat biasa saja, lihat tab Segmentasi Pelanggan untuk profil tiap segmen."]),
            ]),
        ]), className="mb-4 shadow-sm border-start border-4", style={"borderColor": f'{COLORS["teal"]} !important'}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Tingkat Churn per Negara"), dcc.Graph(figure=fig_geo, config={"displayModeBar": False})])), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Komposisi Nasabah: Bertahan vs Churn"), dcc.Graph(figure=fig_donut, config={"displayModeBar": False})])), md=6),
        ], className="g-3"),
    ])

#SEGMENTASI PELANGGAN

def render_tab_segmentation():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Pilih Metode Segmentasi:", className="fw-bold"),
                dcc.Dropdown(
                    id="cluster-method-dropdown",
                    options=[{"label": v, "value": k} for k, v in CLUSTER_METHODS.items()],
                    value="cluster", clearable=False,
                ),
            ], md=6),
            dbc.Col([
                html.Label("Lihat Distribusi Fitur:", className="fw-bold"),
                dcc.Dropdown(
                    id="segment-feature-dropdown",
                    options=[{"label": FEATURE_LABELS_ID[c], "value": c} for c in BASE_COLS],
                    value="Balance", clearable=False,
                ),
            ], md=6),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Peta Segmen Nasabah (Proyeksi PCA 2D)"),
                dcc.Graph(id="cluster-pca-scatter", config={"displayModeBar": False}),
            ])), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Tingkat Churn per Segmen"),
                dcc.Graph(id="cluster-churn-bar", config={"displayModeBar": False}),
            ])), md=5),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Distribusi Fitur per Segmen"),
                dcc.Graph(id="cluster-feature-dist", config={"displayModeBar": False}),
            ])), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Profil Ringkas Tiap Segmen"),
                html.Div(id="cluster-profile-table"),
            ])), md=6),
        ], className="g-3"),
    ])


@app.callback(
    Output("cluster-pca-scatter", "figure"),
    Output("cluster-churn-bar", "figure"),
    Output("cluster-feature-dist", "figure"),
    Output("cluster-profile-table", "children"),
    Input("cluster-method-dropdown", "value"),
    Input("segment-feature-dropdown", "value"),
)
def update_segmentation_tab(cluster_col, feature_col):
    label_map = label_clusters_by_churn(df, cluster_col)
    d = df.copy()
    d["segment_label"] = d[cluster_col].map(label_map)

    order = sorted(d["segment_label"].unique(), key=lambda s: -d[d["segment_label"] == s]["Exited"].mean())

    fig_scatter = px.scatter(
        d, x="PC1", y="PC2", color="segment_label", category_orders={"segment_label": order},
        opacity=0.55, color_discrete_sequence=["#D62839", "#E8871E", "#2EC4B6", "#1B4B66"],
        labels={"segment_label": "Segmen", "PC1": f"PC1 ({PCA_VAR_EXPLAINED[0]:.0%} varians)",
                "PC2": f"PC2 ({PCA_VAR_EXPLAINED[1]:.0%} varians)"},
    )
    fig_scatter.update_traces(marker=dict(size=6))
    fig_scatter.update_layout(template="plotly_white", height=420, legend=dict(orientation="h", y=-0.2),
                               margin=dict(t=10, b=10, l=10, r=10))

    churn_by_seg = d.groupby("segment_label")["Exited"].agg(["mean", "count"]).reindex(order).reset_index()
    fig_bar = px.bar(
        churn_by_seg, x="segment_label", y="mean",
        text=churn_by_seg["mean"].apply(lambda v: f"{v:.1%}"),
        color="mean", color_continuous_scale=["#2EC4B6", "#D62839"],
        labels={"segment_label": "Segmen", "mean": "Tingkat Churn"},
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(template="plotly_white", showlegend=False, coloraxis_showscale=False,
                           yaxis_tickformat=".0%", height=420, margin=dict(t=10, b=10, l=10, r=10),
                           xaxis=dict(tickangle=-10))

    feat_label_col = feature_col + "_label"
    dist = d.groupby(["segment_label", feat_label_col]).size().reset_index(name="count")
    dist["pct"] = dist.groupby("segment_label")["count"].transform(lambda x: x / x.sum())
    fig_dist = px.bar(
        dist, x=feat_label_col, y="pct", color="segment_label", barmode="group",
        category_orders={"segment_label": order},
        color_discrete_sequence=["#D62839", "#E8871E", "#2EC4B6", "#1B4B66"],
        labels={feat_label_col: FEATURE_LABELS_ID[feature_col], "pct": "Proporsi", "segment_label": "Segmen"},
    )
    fig_dist.update_layout(template="plotly_white", yaxis_tickformat=".0%", height=380,
                            legend=dict(orientation="h", y=-0.3), margin=dict(t=10, b=10, l=10, r=10))

    profile_rows = []
    for seg in order:
        sub = d[d["segment_label"] == seg]
        profile_rows.append({
            "Segmen": seg,
            "Jumlah Nasabah": f"{len(sub):,}",
            "Tingkat Churn": f"{sub['Exited'].mean():.1%}",
            "Negara Dominan": sub["Geography_label"].mode()[0],
            "Usia Dominan": sub["Age_label"].mode()[0],
            "Saldo Dominan": sub["Balance_label"].mode()[0],
            "Status Aktif Dominan": sub["IsActiveMember_label"].mode()[0],
        })
    profile_df = pd.DataFrame(profile_rows)
    narrative = html.Div([
        html.P(describe_cluster(d[d["segment_label"] == seg], seg))
        for seg in order
    ], className="mt-2 small text-muted")
    table = dash_table.DataTable(
        data=profile_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in profile_df.columns],
        style_cell={"fontFamily": "Inter, sans-serif", "fontSize": 13, "padding": "8px", "textAlign": "left"},
        style_header={"backgroundColor": COLORS["navy"], "color": "white", "fontWeight": "bold"},
        style_data={"whiteSpace": "normal", "height": "auto"},
        style_table={"overflowX": "auto"},
    )
    return fig_scatter, fig_bar, fig_dist, html.Div([table, narrative])


#PATTERN MINING (ASSOCIATION RULES)

def render_tab_patterns():
    return html.Div([
        dbc.Row([

            dbc.Col([

                html.Label("Negara"),

                dcc.Dropdown(
                    id="country-filter",
                    options=[
                        {"label":"Semua Negara","value":"ALL"}
                    ] + [
                        {"label":x,"value":x}
                        for x in sorted(df["Geography_label"].unique())
                    ],
                    value="ALL",
                    clearable=False,
                )

            ], md=4)

        ], className="mb-3"),

        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Minimum Support", className="fw-bold small"),
                    dcc.Slider(id="min-support-slider", min=0.05, max=0.30, step=0.01, value=0.10,
                               marks={0.05: "0.05", 0.10: "0.10", 0.15: "0.15", 0.20: "0.20", 0.30: "0.30"},
                               tooltip={"placement": "bottom"}),
                ], md=4),
                dbc.Col([
                    html.Label("Minimum Confidence", className="fw-bold small"),
                    dcc.Slider(id="min-confidence-slider", min=0.4, max=0.9, step=0.05, value=0.6,
                               marks={0.4: "0.4", 0.6: "0.6", 0.8: "0.8", 0.9: "0.9"},
                               tooltip={"placement": "bottom"}),
                ], md=4),
                dbc.Col([
                    html.Label("Jumlah Aturan Ditampilkan (Jaringan)", className="fw-bold small"),
                    dcc.Slider(id="top-n-slider", min=5, max=40, step=5, value=20,
                               marks={5: "5", 20: "20", 40: "40"},
                               tooltip={"placement": "bottom"}),
                ], md=4),
            ]),
        ]), className="mb-3 shadow-sm"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Jaringan Aturan Asosiasi (Rule Network)"),
                html.P("Panah menunjukkan arah aturan: A \u2192 B artinya 'jika A maka cenderung B'. "
                       "Ketebalan panah = confidence, transparansi = lift.", className="text-muted small"),
                dcc.Graph(id="rule-network-graph", config={"displayModeBar": False}),
            ])), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Top Aturan berdasarkan Lift"),
                dcc.Graph(id="rule-top-bar", config={"displayModeBar": False}),
            ])), md=5),
        ], className="g-3 mb-3"),

        dbc.Card(dbc.CardBody([
            html.H6("Tabel Lengkap Aturan Asosiasi"),
            html.Div(id="rules-table-container"),
        ]), className="shadow-sm"),
    ])


@app.callback(
    Output("rule-network-graph", "figure"),
    Output("rule-top-bar", "figure"),
    Output("rules-table-container", "children"),
    Input("country-filter", "value"),
    Input("min-support-slider", "value"),
    Input("min-confidence-slider", "value"),
    Input("top-n-slider", "value"),
)
def update_patterns_tab(country, min_support, min_confidence, top_n):
    if country == "ALL":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["Geography_label"] == country]
    rules_df = mine_association_rules(
        filtered_df,
        min_support=min_support,
        min_confidence=min_confidence,
    )
    fig_network = build_rule_network_figure(rules_df, top_n=top_n)

    if rules_df.empty:
        fig_bar = go.Figure()
        fig_bar.update_layout(template="plotly_white", height=480,
                               annotations=[dict(text="Tidak ada aturan ditemukan.", showarrow=False)])
        table = html.P("Tidak ada aturan yang memenuhi ambang batas ini. Coba turunkan nilai minimum.",
                        className="text-muted")
        return fig_network, fig_bar, table

    top10 = rules_df.head(10).copy()
    top10["rule_label"] = top10["antecedent"] + " \u2192 " + top10["consequent"]
    fig_bar = px.bar(
        top10.sort_values("lift"), x="lift", y="rule_label", orientation="h",
        color="confidence", color_continuous_scale=["#1B4B66", "#2EC4B6"],
        labels={"lift": "Lift", "rule_label": "", "confidence": "Confidence"},
    )
    fig_bar.update_layout(template="plotly_white", height=480, margin=dict(l=10, r=10, t=10, b=10))

    display_df = rules_df.head(50).copy()
    display_df["support"] = display_df["support"].round(3)
    display_df["confidence"] = display_df["confidence"].round(3)
    display_df["lift"] = display_df["lift"].round(3)
    display_df.columns = ["Jika (Antecedent)", "Maka (Consequent)", "Support", "Confidence", "Lift"]

    insight_sentences = rules_to_business_insights(rules_df, top_n=10)
    insight_block = html.Div([
        html.H6("Insight Bisnis dari 10 Rule Teratas", className="mt-3"),
        html.Ol([html.Li(s) for s in insight_sentences]),
    ])

    table = dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display_df.columns],
        page_size=10, sort_action="native", filter_action="native",
        style_cell={"fontFamily": "Inter, sans-serif", "fontSize": 13, "padding": "8px", "textAlign": "left"},
        style_header={"backgroundColor": COLORS["navy"], "color": "white", "fontWeight": "bold"},
        style_table={"overflowX": "auto"},
    )
    return fig_network, fig_bar, html.Div([table, insight_block])



DEVIATION_THRESHOLD = 0.15
DEV_COUNT_CUTOFF = 3

ANOMALY_TYPE_DESC = {
    "Anomali Global": (
        "Profil nasabah ini jarang ditemukan di seluruh populasi, bukan cuma di segmennya. "
        "Layak jadi prioritas investigasi manual karena polanya benar-benar tidak lazim."
    ),
    "Anomali Kontekstual (dalam Segmen)": (
        "Secara umum profil ini biasa saja, tapi tidak lazim dibanding nasabah lain di segmen "
        "yang sama. Bisa jadi tanda nasabah ini sebenarnya salah masuk segmen, atau punya "
        "kebutuhan berbeda dari teman-teman segmennya."
    ),
    "Anomali Kombinasi Fitur": (
        "Tidak ada satu fitur pun yang ekstrem sendirian, tapi kombinasi beberapa fitur "
        "sekaligus jarang terjadi bersamaan. Sinyal ini baru terlihat lewat model multivariat "
        "(Isolation Forest), tidak akan ketahuan hanya dengan melihat satu kolom pada satu waktu."
    ),
}

FEATURE_BUSINESS_NOTE = {
    "CreditScore": "Skor kredit yang tidak lazim untuk profil ini bisa berarti risiko "
                   "kredit yang perlu ditinjau ulang (jika rendah), atau nasabah kualitas "
                   "tinggi yang perlu dijaga (jika tinggi).",
    "Geography": "Kombinasi negara yang jarang untuk profil ini bisa menandakan preferensi "
                 "atau kebutuhan layanan regional yang belum terlayani dengan baik.",
    "Gender": "Perbedaan pola berdasarkan gender jarang jadi indikator risiko yang berdiri "
              "sendiri; biasanya baru bermakna jika berinteraksi dengan fitur lain.",
    "Age": "Usia yang tidak lazim untuk kombinasi produk/saldo nasabah ini bisa menandakan "
           "produk yang ditawarkan belum sesuai dengan tahap hidupnya.",
    "Tenure": "Lama menjadi nasabah yang tidak lazim (baru sekali atau justru sangat lama) "
              "bisa menandakan masalah onboarding di awal, atau kejenuhan layanan pada "
              "nasabah loyal jangka panjang.",
    "Balance": "Saldo pada level yang jarang ditemui sering menandakan nasabah bernilai "
               "tinggi (flight risk) yang layak mendapat perhatian relationship manager.",
    "NumOfProducts": "Jumlah produk yang tidak lazim membuka peluang cross-sell (jika hanya "
                      "1 produk) atau perlu ditinjau potensi overexposure-nya (jika sangat banyak).",
    "HasCrCard": "Kepemilikan kartu kredit yang tidak lazim untuk profil ini jarang jadi "
                 "pendorong utama churn, tapi tetap relevan untuk penargetan penawaran produk.",
    "IsActiveMember": "Status keaktifan yang tidak lazim untuk profil ini adalah sinyal kuat "
                       "untuk kampanye reaktivasi atau program engagement khusus.",
    "EstimatedSalary": "Estimasi gaji pada level yang jarang ditemui membuka peluang "
                        "penargetan produk yang lebih sesuai dengan tingkat pendapatan nasabah.",
}


def _build_anomaly_diagnostics(data, base_cols, threshold=DEVIATION_THRESHOLD, top_k=2):
    """Untuk tiap nasabah anomali: hitung seberapa jarang tiap fitur dibanding
    (1) seluruh populasi normal, dan (2) populasi normal di segmen yang sama.
    Lalu klasifikasikan jenis anomalinya dan simpan fitur-fitur paling jarang."""
    normal = data[data["Anomaly"] == "Normal"]
    anomalies = data[data["Anomaly"] == "Anomaly"]

    global_freq = {col: normal[col + "_label"].value_counts(normalize=True).to_dict() for col in base_cols}
    cluster_freq = {}
    for cl, sub in normal.groupby("cluster"):
        cluster_freq[cl] = {col: sub[col + "_label"].value_counts(normalize=True).to_dict() for col in base_cols}

    records = []
    for idx, row in anomalies.iterrows():
        cl = row["cluster"]
        feat_scores = []
        global_dev, contextual_dev = 0, 0
        for col in base_cols:
            val = row[col + "_label"]
            g = global_freq[col].get(val, 0.0)
            c = cluster_freq.get(cl, {}).get(col, {}).get(val, 0.0)
            feat_scores.append((col, val, g, c))
            if g < threshold:
                global_dev += 1
            if c < threshold:
                contextual_dev += 1

        if global_dev >= DEV_COUNT_CUTOFF:
            anomaly_type = "Anomali Global"
        elif contextual_dev >= DEV_COUNT_CUTOFF:
            anomaly_type = "Anomali Kontekstual (dalam Segmen)"
        else:
            anomaly_type = "Anomali Kombinasi Fitur"

        feat_scores.sort(key=lambda x: min(x[2], x[3]))
        top_feats = feat_scores[:top_k]

        factor_parts, insight_parts = [], []
        for col, val, g, c in top_feats:
            fname = FEATURE_LABELS_ID[col]
            if g <= c:
                ctx = f"hanya {g:.0%} nasabah normal secara keseluruhan"
            else:
                ctx = f"hanya {c:.0%} nasabah normal di segmen yang sama"
            factor_parts.append(f"{fname} = {val} ({ctx})")
            insight_parts.append(FEATURE_BUSINESS_NOTE.get(col, ""))

        records.append({
            "Jenis Anomali": anomaly_type,
            "Faktor Utama": "; ".join(factor_parts),
            "Insight Bisnis": " ".join(dict.fromkeys(insight_parts)),  # dedupe, keep order
            "Status Churn": row["Exited_label"],
            "_severity": global_dev + contextual_dev,
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("_severity", ascending=False).drop(columns="_severity").reset_index(drop=True)
    return result


_anomaly_diagnostics = _build_anomaly_diagnostics(df, BASE_COLS)
_anomaly_type_counts = _anomaly_diagnostics["Jenis Anomali"].value_counts().to_dict() if not _anomaly_diagnostics.empty else {}


def _build_anomaly_type_bar():
    if _anomaly_diagnostics.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=280,
                           annotations=[dict(text="Tidak ada data anomali.", showarrow=False)])
        return fig
    counts = _anomaly_diagnostics["Jenis Anomali"].value_counts().reset_index()
    counts.columns = ["Jenis Anomali", "Jumlah"]
    order = ["Anomali Global", "Anomali Kontekstual (dalam Segmen)", "Anomali Kombinasi Fitur"]
    counts["Jenis Anomali"] = pd.Categorical(counts["Jenis Anomali"], categories=order, ordered=True)
    counts = counts.sort_values("Jenis Anomali")
    fig = px.bar(
        counts, x="Jenis Anomali", y="Jumlah", text="Jumlah", color="Jenis Anomali",
        color_discrete_map={
            "Anomali Global": "#D62839",
            "Anomali Kontekstual (dalam Segmen)": "#E8871E",
            "Anomali Kombinasi Fitur": "#1B4B66",
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_white", showlegend=False, height=300,
                       margin=dict(t=20, b=20, l=20, r=20), xaxis_title="", yaxis_title="Jumlah Nasabah Anomali")
    return fig

#ANOMALI

def render_tab_anomaly():
    fig_scatter = px.scatter(
        df, x="PC1", y="PC2", color="Anomaly",
        color_discrete_map={"Normal": "#2EC4B6", "Anomaly": "#D62839"},
        opacity=0.55,
        labels={"PC1": f"PC1 ({PCA_VAR_EXPLAINED[0]:.0%} varians)", "PC2": f"PC2 ({PCA_VAR_EXPLAINED[1]:.0%} varians)"},
    )
    fig_scatter.update_traces(marker=dict(size=6))
    fig_scatter.update_layout(template="plotly_white", height=420, legend=dict(orientation="h", y=-0.2),
                               margin=dict(t=10, b=10, l=10, r=10))

    cross = pd.crosstab(df["cluster"], df["Anomaly"])
    fig_cross = go.Figure(data=[
        go.Bar(name="Normal", x=[f"Klaster {i}" for i in cross.index], y=cross["Normal"], marker_color="#2EC4B6"),
        go.Bar(name="Anomaly", x=[f"Klaster {i}" for i in cross.index], y=cross["Anomaly"], marker_color="#D62839"),
    ])
    fig_cross.update_layout(barmode="stack", template="plotly_white", height=420,
                             legend=dict(orientation="h", y=-0.2), margin=dict(t=10, b=10, l=10, r=10),
                             yaxis_title="Jumlah Nasabah")

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Anomali Terdeteksi", f"{ANOMALY_COUNT:,}", f"{ANOMALY_RATE:.1%} dari total nasabah", COLORS["amber"]), md=3),
            dbc.Col(kpi_card("Churn Rate (Anomaly)", f"{CHURN_RATE_ANOMALY:.1%}", "vs " + f"{CHURN_RATE_NORMAL:.1%} pada nasabah normal", COLORS["danger"]), md=3),
            dbc.Col(kpi_card("Outlier DBSCAN", f"{OUTLIER_DBSCAN_COUNT:,}", "di luar kepadatan klaster utama", COLORS["navy2"]), md=3),
            dbc.Col(kpi_card("Churn Rate (Outlier DBSCAN)", f"{CHURN_RATE_DBSCAN_OUTLIER:.1%}", "vs " + f"{CHURN_RATE_DBSCAN_NORMAL:.1%} pada klaster utama", COLORS["danger"]), md=3),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Peta Anomali (Isolation Forest, Proyeksi PCA)"),
                dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}),
            ])), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Anomali vs Klaster Segmentasi"),
                dcc.Graph(figure=fig_cross, config={"displayModeBar": False}),
            ])), md=6),
        ], className="g-3 mb-3"),

        dbc.Card(dbc.CardBody([
            html.H6("Distribusi Fitur: Normal vs Anomaly"),
            html.Label("Pilih Fitur:", className="fw-bold small"),
            dcc.Dropdown(
                id="anomaly-feature-dropdown",
                options=[{"label": FEATURE_LABELS_ID[c], "value": c} for c in BASE_COLS],
                value="Balance", clearable=False, style={"maxWidth": "400px"},
            ),
            dcc.Graph(id="anomaly-feature-box", config={"displayModeBar": False}),
        ]), className="shadow-sm"),

        dbc.Card(dbc.CardBody([
            html.H6("Kenapa Nasabah Ini Dianggap Anomali?"),
            html.P(
                "Setiap nasabah anomali dikelompokkan ke salah satu dari 3 jenis, tergantung "
                "seberapa jarang profilnya dibandingkan populasi keseluruhan vs. dibandingkan "
                "segmen tempatnya berada:",
                className="text-muted small mb-2",
            ),
            html.Ul([
                html.Li([html.B("Anomali Global: "), ANOMALY_TYPE_DESC["Anomali Global"]]),
                html.Li([html.B("Anomali Kontekstual (dalam Segmen): "),
                         ANOMALY_TYPE_DESC["Anomali Kontekstual (dalam Segmen)"]]),
                html.Li([html.B("Anomali Kombinasi Fitur: "), ANOMALY_TYPE_DESC["Anomali Kombinasi Fitur"]]),
            ], className="small mb-3"),
            dbc.Alert(
                "Catatan: \u2018Anomali Kolektif\u2019 (collective anomaly) tidak dipakai di sini karena "
                "tiap baris adalah nasabah independen, bukan data runtun waktu \u2014 konsep "
                "\u2018sekumpulan titik yang anomali bersama\u2019 tidak relevan untuk struktur data ini.",
                color="light", className="small border mb-3",
            ),

            dcc.Graph(figure=_build_anomaly_type_bar(), config={"displayModeBar": False}),

            html.H6("Rincian per Nasabah (Faktor Utama & Insight Bisnis)", className="mt-3"),
            html.P(
                "Diurutkan dari yang paling menyimpang. \u2018Faktor Utama\u2019 menunjukkan 2 fitur "
                "paling langka pada nasabah tersebut, lengkap dengan seberapa langka nilainya.",
                className="text-muted small",
            ),
            dash_table.DataTable(
                data=_anomaly_diagnostics.to_dict("records"),
                columns=[{"name": c, "id": c} for c in _anomaly_diagnostics.columns],
                page_size=10, sort_action="native", filter_action="native",
                style_cell={"fontFamily": "Inter, sans-serif", "fontSize": 12.5, "padding": "8px",
                            "textAlign": "left", "whiteSpace": "normal", "height": "auto"},
                style_header={"backgroundColor": COLORS["navy"], "color": "white", "fontWeight": "bold"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Jenis Anomali} = "Anomali Global"'},
                     "backgroundColor": "#FDECEC"},
                    {"if": {"filter_query": '{Jenis Anomali} = "Anomali Kontekstual (dalam Segmen)"'},
                     "backgroundColor": "#FFF3E0"},
                ],
                style_table={"overflowX": "auto"},
                style_cell_conditional=[
                    {"if": {"column_id": "Faktor Utama"}, "minWidth": "260px"},
                    {"if": {"column_id": "Insight Bisnis"}, "minWidth": "320px"},
                ],
            ),
        ]), className="shadow-sm mt-3"),
    ])


@app.callback(
    Output("anomaly-feature-box", "figure"),
    Input("anomaly-feature-dropdown", "value"),
)
def update_anomaly_feature(feature_col):
    feat_label_col = feature_col + "_label"
    dist = df.groupby(["Anomaly", feat_label_col]).size().reset_index(name="count")
    dist["pct"] = dist.groupby("Anomaly")["count"].transform(lambda x: x / x.sum())
    fig = px.bar(
        dist, x=feat_label_col, y="pct", color="Anomaly", barmode="group",
        color_discrete_map={"Normal": "#2EC4B6", "Anomaly": "#D62839"},
        labels={feat_label_col: FEATURE_LABELS_ID[feature_col], "pct": "Proporsi"},
    )
    fig.update_layout(template="plotly_white", yaxis_tickformat=".0%", height=380,
                       legend=dict(orientation="h", y=-0.25), margin=dict(t=10, b=10, l=10, r=10))
    return fig


#KNOWLEDGE DISCOVERY REPORT

def render_tab_report():
    top_products_rule = _default_rules[
        (_default_rules["consequent"].str.contains("Produk", na=False)) &
        (_default_rules["antecedent"].str.contains("Churn", na=False))
    ]
    top_active_rule = _default_rules[
        (_default_rules["consequent"].str.contains("Keaktifan", na=False)) &
        (_default_rules["antecedent"].str.contains("Churn", na=False))
    ]

    product_line = "Data pola tidak cukup pada ambang default untuk merangkum baris ini secara otomatis lihat tab Pattern Mining."
    if not top_products_rule.empty:
        r = top_products_rule.iloc[0]
        product_line = (f"Nasabah yang churn memiliki pola \u201c{r['antecedent']} \u2192 {r['consequent']} "
                         f"dengan confidence {r['confidence']:.0%} dan lift {r['lift']:.2f}.")

    active_line = None
    if not top_active_rule.empty:
        r = top_active_rule.iloc[0]
        active_line = (f"Pola \u201c{r['antecedent']} \u2192 {r['consequent']}\u201d muncul dengan confidence "
                        f"{r['confidence']:.0%} (lift {r['lift']:.2f}).")

    return html.Div([
        dbc.Card(dbc.CardBody([
            html.H4("Knowledge Discovery Report", className="mb-1"),
            html.P("Bank Customer Churn: Ringkasan Temuan pada sisi Bisnis", className="text-muted mb-4"),

            html.H5("1. Ringkasan Eksekutif"),
            html.P(f"Dari {N_TOTAL:,} nasabah yang dianalisis, {CHURN_RATE:.1%} di antaranya telah berhenti "
                   "menggunakan layanan bank (churn). Analisis ini tidak berhenti pada angka tersebut dengan "
                   "menggabungkan segmentasi nasabah, pola asosiasi, dan deteksi anomali, kami menemukan sinyal "
                   "risiko yang tidak terlihat jika hanya melihat rata-rata atau tabel data mentah."),

            html.H5("2. Jawaban atas Pertanyaan Sentral", className="mt-4"),
            html.P(html.I("\u201cApa yang kita temukan yang tidak terlihat jelas dari data mentah?\u201d")),
            html.Ol([
                html.Li([html.B("Nasabah tidak biasa = risiko tinggi. "),
                         f"Nasabah yang secara statistik menyimpang dari pola mayoritas (terdeteksi baik oleh DBSCAN "
                         f"maupun Isolation Forest) memiliki tingkat churn masing-masing {CHURN_RATE_DBSCAN_OUTLIER:.1%} "
                         f"dan {CHURN_RATE_ANOMALY:.1%} sekitar 2\u20133x lebih tinggi dibanding nasabah pada pola "
                         "umum. Ini adalah sinyal peringatan dini yang mustahil terlihat tanpa analisis kelompok."]),
                html.Li([html.B("Sebagian besar anomali bukan karena satu fitur ekstrem, tapi kombinasi. "),
                         f"Dari {ANOMALY_COUNT} nasabah anomali, {_anomaly_type_counts.get('Anomali Kombinasi Fitur', 0)} "
                         "di antaranya ('Anomali Kombinasi Fitur') terlihat normal di setiap kolom jika dilihat satu "
                         "per satu \u2014 keanehannya baru muncul saat beberapa fitur digabungkan. Ini bukti bahwa "
                         "meninjau data kolom-per-kolom saja tidak cukup untuk menangkap risiko tersembunyi (lihat "
                         "tab Deteksi Anomali)."]),
                html.Li([html.B("Segmentasi mengungkap kelompok nasabah dengan wajah risiko berbeda "),
                         "walau rata-rata keseluruhan terlihat homogen. Segmen tertentu (lihat tab Segmentasi) "
                         "menunjukkan tingkat churn jauh di atas segmen lain, membuka peluang strategi retensi "
                         "yang ditargetkan, bukan pukul rata ke semua nasabah."]),
                html.Li([html.B("Kombinasi karakteristik, bukan satu variabel tunggal, yang mendorong churn. "),
                         product_line]),
            ] + ([html.Li([html.B("Keaktifan nasabah adalah sinyal kuat. "), active_line])] if active_line else [])),

            html.H5("3. Rekomendasi Tindakan Bisnis", className="mt-4"),
            html.Ul([
                html.Li("Bangun program monitoring khusus untuk nasabah yang ditandai sebagai \u2018anomali\u2019/\u2018outlier\u2019 "
                        "oleh sistem mereka adalah kandidat prioritas untuk intervensi retensi dini."),
                html.Li("Rancang penawaran cross-sell (produk kedua) yang ditargetkan pada segmen dengan tingkat "
                        "penggunaan 1 produk yang tinggi, karena berasosiasi kuat dengan churn."),
                html.Li("Aktifkan kembali nasabah tidak aktif melalui kampanye reaktivasi, mengingat keterkaitannya "
                        "dengan risiko churn."),
                html.Li("Gunakan profil segmen (tab Segmentasi Pelanggan) untuk menyesuaikan pesan pemasaran dan "
                        "layanan per kelompok, alih-alih strategi satu-untuk-semua."),
            ]),

            html.H5("4. Catatan Metodologi", className="mt-4"),
            html.P("Segmentasi menggunakan K-Modes, K-Means, dan DBSCAN untuk saling memvalidasi kelompok nasabah. "
                   "Pola asosiasi ditemukan dengan algoritma Apriori (support & confidence dapat diatur pada tab "
                   "Pattern Mining). Anomali dideteksi dengan Isolation Forest (asumsi 5% data adalah anomali) dan "
                   "divalidasikan silang dengan hasil DBSCAN. Keberhasilan analisis ini diukur dari seberapa "
                   "actionable dan non-trivial temuannya bagi tim bisnis \u2014 bukan dari akurasi model semata."),

        ]), className="shadow-sm"),
    ])


#APP LAYOUT (NAVBAR + TABS)

app.layout = html.Div([
    build_navbar(),

    html.Div([
        dcc.Tabs(id="main-tabs", value="tab-overview", children=[
            dcc.Tab(label="Ringkasan Eksekutif", value="tab-overview"),
            dcc.Tab(label="Segmentasi Pelanggan", value="tab-segmentation"),
            dcc.Tab(label="Pattern Mining", value="tab-patterns"),
            dcc.Tab(label="Deteksi Anomali", value="tab-anomaly"),
            dcc.Tab(label="Knowledge Discovery Report", value="tab-report"),
        ], className="custom-tabs"),
        html.Div(id="tab-content", className="tab-content-wrap"),
    ], className="app-body"),
], className="app-container")


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab_value):
    if tab_value == "tab-overview":
        return render_tab_overview()
    elif tab_value == "tab-segmentation":
        return render_tab_segmentation()
    elif tab_value == "tab-patterns":
        return render_tab_patterns()
    elif tab_value == "tab-anomaly":
        return render_tab_anomaly()
    elif tab_value == "tab-report":
        return render_tab_report()
    return html.Div("Tab tidak ditemukan.")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)