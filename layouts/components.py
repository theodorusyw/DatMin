import dash_bootstrap_components as dbc
from dash import html

from core.constants import COLORS

def kpi_card(title, value, subtitle, color=COLORS["navy2"]):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value", style={"color": color}),
            html.Div(subtitle, className="kpi-subtitle"),
        ]),
        className="kpi-card shadow-sm",
    )

def build_navbar(title="Bank Customer Insight Dashboard",
                  subtitle="Phase 5 — Visualization & Knowledge Presentation"):
    """Navbar utama aplikasi. Dipisah dari app.layout supaya title/subtitle
    bisa di-reuse atau di-parameterize kalau dashboard ini di-clone."""
    return html.Div(
        html.Div([
            html.Div(title, className="navbar-title"),
            html.Div(subtitle, className="navbar-subtitle"),
        ], className="navbar-brand-wrap"),
        className="app-navbar",
    )