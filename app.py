import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Backtest Trade Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Backtest Trade Analyzer")
st.write(
    "Upload a backtest CSV with `Exit Date` and `Pnl` columns to review equity, drawdown, distribution, and performance metrics."
)


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_cols = {col.lower(): col for col in df.columns}
    if "exit date" not in lower_cols or "pnl" not in lower_cols:
        st.error("CSV must contain columns named `Exit Date` and `Pnl`.")
        return pd.DataFrame()
    df = df.rename(
        columns={lower_cols["exit date"]: "Exit Date", lower_cols["pnl"]: "Pnl"}
    )
    return df


@st.cache_data
def parse_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = validate_columns(df)
    if df.empty:
        return df

    df["Exit Date"] = pd.to_datetime(df["Exit Date"], errors="coerce")
    df["Pnl"] = pd.to_numeric(df["Pnl"], errors="coerce")
    df = df.dropna(subset=["Exit Date", "Pnl"])
    df = df.sort_values("Exit Date").reset_index(drop=True)

    df["Equity"] = df["Pnl"].cumsum()
    df["High Water Mark"] = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] - df["High Water Mark"]
    df["Drawdown %"] = np.where(
        df["High Water Mark"] > 0,
        df["Drawdown"] / df["High Water Mark"],
        0,
    )
    df["Year"] = df["Exit Date"].dt.year
    df["Month"] = df["Exit Date"].dt.to_period("M").astype(str)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    total_trades = len(df)
    winning_trades = df[df["Pnl"] > 0]
    losing_trades = df[df["Pnl"] <= 0]
    win_rate = len(winning_trades) / total_trades if total_trades else 0
    avg_win = winning_trades["Pnl"].mean() if len(winning_trades) else 0
    avg_loss = losing_trades["Pnl"].mean() if len(losing_trades) else 0
    avg_loss_abs = abs(avg_loss) if avg_loss != 0 else 0
    profit_factor = (
        winning_trades["Pnl"].sum() / abs(losing_trades["Pnl"].sum())
        if abs(losing_trades["Pnl"].sum()) > 0
        else np.nan
    )
    risk_reward = avg_win / avg_loss_abs if avg_loss_abs != 0 else np.nan
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    pnl_std = df["Pnl"].std()
    sharpe = df["Pnl"].mean() / pnl_std if pnl_std and pnl_std != 0 else np.nan
    max_drawdown = df["Drawdown"].min() if "Drawdown" in df else 0
    cumulative_pnl = df["Equity"].iloc[-1] if total_trades else 0
    largest_win = winning_trades["Pnl"].max() if len(winning_trades) else 0
    largest_loss = losing_trades["Pnl"].min() if len(losing_trades) else 0

    start_date = df["Exit Date"].min() if total_trades else None
    end_date = df["Exit Date"].max() if total_trades else None
    years = (end_date - start_date).days / 365.25 if start_date is not None else 0
    annualized_pnl = cumulative_pnl / years if years > 0 else np.nan

    return {
        "Total Trades": total_trades,
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate": win_rate,
        "Average Win": avg_win,
        "Average Loss": avg_loss,
        "Risk/Reward": risk_reward,
        "Profit Factor": profit_factor,
        "Expectancy": expectancy,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Cumulative PnL": cumulative_pnl,
        "Annualized PnL": annualized_pnl,
        "Largest Win": largest_win,
        "Largest Loss": largest_loss,
        "Start Date": start_date,
        "End Date": end_date,
    }


def plot_equity_curve(df: pd.DataFrame):
    fig = px.line(
        df,
        x="Exit Date",
        y="Equity",
        title="Equity Curve",
        labels={"Exit Date": "Exit Date", "Equity": "Equity"},
    )
    fig.update_traces(mode="lines+markers", marker=dict(size=4), line=dict(width=2))
    return fig


def plot_drawdown_curve(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Exit Date"],
            y=df["Drawdown"],
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#EF553B"),
        )
    )
    fig.update_layout(
        title="Drawdown Curve",
        xaxis_title="Exit Date",
        yaxis_title="Drawdown",
    )
    return fig


def plot_yearly_pnl(df: pd.DataFrame):
    yearly = df.groupby("Year")["Pnl"].sum().reset_index()
    fig = px.bar(
        yearly,
        x="Year",
        y="Pnl",
        title="Yearly PnL Distribution",
        labels={"Pnl": "Total PnL"},
        color="Pnl",
        color_continuous_scale="Tealrose",
    )
    return fig


def plot_monthly_pnl(df: pd.DataFrame):
    monthly = df.groupby("Month")["Pnl"].sum().reset_index()
    fig = px.bar(
        monthly,
        x="Month",
        y="Pnl",
        title="Monthly PnL Distribution",
        labels={"Pnl": "Total PnL", "Month": "Month"},
        color="Pnl",
        color_continuous_scale="Blues",
    )
    fig.update_xaxes(tickangle=45)
    return fig


uploaded_file = st.file_uploader(
    "Upload backtest CSV",
    type=["csv"],
    help="CSV must contain `Exit Date` and `Pnl` columns.",
)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    df = parse_data(raw_df)

    if df.empty:
        st.stop()

    st.markdown("### Sample of the imported data")
    st.dataframe(df[["Exit Date", "Pnl"]].head(10), use_container_width=True)

    metrics = compute_metrics(df)

    st.markdown("### Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", metrics["Total Trades"])
    col2.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
    col3.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}" if not np.isnan(metrics["Profit Factor"]) else "N/A")
    col4.metric("Risk / Reward", f"{metrics['Risk/Reward']:.2f}" if not np.isnan(metrics["Risk/Reward"]) else "N/A")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Cumulative PnL", f"{metrics['Cumulative PnL']:.2f}")
    col6.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2f}")
    col7.metric("Average Win", f"{metrics['Average Win']:.2f}")
    col8.metric("Average Loss", f"{metrics['Average Loss']:.2f}")

    st.markdown("### Additional Trade Statistics")
    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Largest Win", f"{metrics['Largest Win']:.2f}")
    col10.metric("Largest Loss", f"{metrics['Largest Loss']:.2f}")
    col11.metric("Expectancy", f"{metrics['Expectancy']:.2f}")
    col12.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}" if not np.isnan(metrics["Sharpe Ratio"]) else "N/A")

    st.markdown("### Visual Analysis")
    eq_col, dd_col = st.columns(2)
    eq_col.plotly_chart(plot_equity_curve(df), use_container_width=True)
    dd_col.plotly_chart(plot_drawdown_curve(df), use_container_width=True)

    st.plotly_chart(plot_yearly_pnl(df), use_container_width=True)
    st.plotly_chart(plot_monthly_pnl(df), use_container_width=True)

    st.markdown("### Detailed Backtest Report")
    report = {
        "Start Date": metrics["Start Date"].strftime("%Y-%m-%d") if metrics["Start Date"] is not None else "N/A",
        "End Date": metrics["End Date"].strftime("%Y-%m-%d") if metrics["End Date"] is not None else "N/A",
        "Total Trades": metrics["Total Trades"],
        "Winning Trades": metrics["Winning Trades"],
        "Losing Trades": metrics["Losing Trades"],
        "Win Rate": f"{metrics['Win Rate']:.2%}",
        "Profit Factor": f"{metrics['Profit Factor']:.2f}" if not np.isnan(metrics["Profit Factor"]) else "N/A",
        "Risk / Reward": f"{metrics['Risk/Reward']:.2f}" if not np.isnan(metrics["Risk/Reward"]) else "N/A",
        "Annualized PnL": f"{metrics['Annualized PnL']:.2f}" if not np.isnan(metrics["Annualized PnL"]) else "N/A",
        "Max Drawdown": f"{metrics['Max Drawdown']:.2f}",
    }
    st.table(pd.DataFrame(report, index=[0]).T.rename(columns={0: "Value"}))

    if st.checkbox("Show full parsed data", value=False):
        st.dataframe(df, use_container_width=True)
else:
    st.info("Upload a CSV file with `Exit Date` and `Pnl` to begin.")
