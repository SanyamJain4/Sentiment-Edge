"""
Reusable Plotly chart-builders for the Nifty 50 pipeline.

Every function takes plain DataFrames/Series (already filtered to one
ticker where relevant) and returns a `plotly.graph_objects.Figure` —
kept framework-agnostic so the same functions work from a notebook,
a script, or the Streamlit dashboard (`streamlit_app.py`).
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Price chart with technical overlay + up/down signal markers -----------
def plot_price_with_signals(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Price chart with SMA/EMA/Bollinger Bands, plus green/red markers
    showing actual next-day direction (Stock_Movement) — this is the
    "up or down signalling" view requested for the dashboard.
    """
    df = df.sort_values("Date")

    fig = go.Figure()

    # Bollinger band shading
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Upper_Band"], line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Lower_Band"], line=dict(width=0),
        fill="tonexty", fillcolor="rgba(100,100,255,0.08)",
        name="Bollinger Band", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close",
                              line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50",
                              line=dict(color="orange", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_50"], name="EMA 50",
                              line=dict(color="purple", width=1, dash="dot")))

    # Up/down signal markers, placed just below/above the close price
    if "Stock_Movement" in df.columns:
        up_days = df[df["Stock_Movement"] == 1]
        down_days = df[df["Stock_Movement"] == 0]

        fig.add_trace(go.Scatter(
            x=up_days["Date"], y=up_days["Close"] * 0.985,
            mode="markers", name="Up signal",
            marker=dict(symbol="triangle-up", color="green", size=7),
        ))
        fig.add_trace(go.Scatter(
            x=down_days["Date"], y=down_days["Close"] * 1.015,
            mode="markers", name="Down signal",
            marker=dict(symbol="triangle-down", color="red", size=7),
        ))

    fig.update_layout(
        title=f"{ticker} — Price with Technical Overlay & Direction Signals",
        xaxis_title="Date", yaxis_title="Price (INR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
    )
    return fig


# --- MACD with buy/sell crossover markers -----------------------------------
def plot_macd_with_signals(df: pd.DataFrame, ticker: str) -> go.Figure:
    df = df.sort_values("Date").copy()
    df["Hist"] = df["MACD"] - df["MACD_Signal"]

    # Buy = MACD crosses above signal; Sell = MACD crosses below signal
    df["Prev_Hist"] = df["Hist"].shift(1)
    buys = df[(df["Prev_Hist"] < 0) & (df["Hist"] >= 0)]
    sells = df[(df["Prev_Hist"] > 0) & (df["Hist"] <= 0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Date"], y=df["Hist"], name="Histogram",
                          marker_color=["green" if v >= 0 else "red" for v in df["Hist"]],
                          opacity=0.4))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD",
                              line=dict(color="#1f77b4", width=1.5)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal",
                              line=dict(color="orange", width=1.5)))

    fig.add_trace(go.Scatter(
        x=buys["Date"], y=buys["MACD"], mode="markers", name="Buy signal",
        marker=dict(symbol="triangle-up", color="green", size=10, line=dict(width=1, color="black")),
    ))
    fig.add_trace(go.Scatter(
        x=sells["Date"], y=sells["MACD"], mode="markers", name="Sell signal",
        marker=dict(symbol="triangle-down", color="red", size=10, line=dict(width=1, color="black")),
    ))

    fig.update_layout(
        title=f"{ticker} — MACD with Buy/Sell Crossover Signals",
        xaxis_title="Date", yaxis_title="MACD",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=350,
    )
    return fig


# --- RSI with overbought/oversold zones --------------------------------------
def plot_rsi(df: pd.DataFrame, ticker: str) -> go.Figure:
    df = df.sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI",
                              line=dict(color="#1f77b4", width=1.5)))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.05, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.05, line_width=0)

    fig.update_layout(
        title=f"{ticker} — RSI (14-day)",
        xaxis_title="Date", yaxis_title="RSI", yaxis_range=[0, 100],
        height=300,
    )
    return fig


# --- Sentiment over time ------------------------------------------------------
def plot_sentiment_timeline(df: pd.DataFrame, ticker: str) -> go.Figure:
    df = df.sort_values("Date")
    colors = ["green" if v >= 0 else "red" for v in df["sentiment_score"].fillna(0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Date"], y=df["sentiment_score"], marker_color=colors, name="Sentiment"))
    fig.add_hline(y=0, line_color="gray", line_width=1)

    fig.update_layout(
        title=f"{ticker} — News Sentiment Score (FinBERT)",
        xaxis_title="Date", yaxis_title="Sentiment (-1 to 1)",
        height=300,
    )
    return fig


# --- Model performance: confusion matrix -------------------------------------
def plot_confusion_matrix(cm, labels=("Down", "Up")) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
        colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(title="Confusion Matrix (Test Set)", height=350)
    return fig


# --- Model performance: feature importance -----------------------------------
def plot_feature_importance(importances: pd.Series, top_n: int = 15) -> go.Figure:
    top = importances.sort_values(ascending=True).tail(top_n)
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color="#1f77b4"))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances (LightGBM gain)",
        xaxis_title="Importance", height=max(300, top_n * 25),
    )
    return fig


# --- Prediction confidence over time (test period) ---------------------------
def plot_prediction_confidence(predictions_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Shows the model's predicted up-probability vs. what actually happened,
    for a single ticker's test-period rows — makes it easy to see where the
    model was confidently right or confidently wrong."""
    df = predictions_df[predictions_df["Ticker"] == ticker].sort_values("Date")
    correct = df["Predicted"] == df["Actual"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Predicted_Up_Probability"], mode="lines+markers",
        name="Predicted up-probability",
        marker=dict(color=["green" if c else "red" for c in correct], size=7),
        line=dict(color="gray", width=1),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"{ticker} — Predicted Up-Probability vs. Outcome (green=correct, red=wrong)",
        xaxis_title="Date", yaxis_title="Predicted P(Up)", yaxis_range=[0, 1],
        height=350,
    )
    return fig
