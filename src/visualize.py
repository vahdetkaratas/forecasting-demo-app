from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Aligned with vercel_demo/static/index.html :root
_BG = "#0f1115"
_SURFACE = "#181b22"
_BORDER = "#2a2f3a"
_TEXT = "#e6e8ec"
_MUTED = "#8b919e"
_ACCENT = "#818cf8"


def build_forecast_figure(history: pd.DataFrame, future: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["value"],
            mode="lines",
            name="Historical",
            line=dict(color=_MUTED, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future["date"],
            y=future["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color=_ACCENT, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future["date"],
            y=future["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future["date"],
            y=future["yhat_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Confidence interval",
            fillcolor="rgba(99, 102, 241, 0.22)",
            hoverinfo="skip",
        )
    )
    grid = "rgba(255,255,255,0.06)"
    fig.update_layout(
        title=dict(text="Historical vs forecast", font=dict(size=15, color=_TEXT)),
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Series",
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="DM Sans, system-ui, sans-serif", size=12),
        legend=dict(
            bgcolor="rgba(24,27,34,0.9)",
            bordercolor=_BORDER,
            borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(l=24, r=24, t=48, b=24),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        linecolor=_BORDER,
        tickfont=dict(color=_MUTED),
        title_font=dict(color=_MUTED),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        linecolor=_BORDER,
        tickfont=dict(color=_MUTED),
        title_font=dict(color=_MUTED),
    )
    return fig
