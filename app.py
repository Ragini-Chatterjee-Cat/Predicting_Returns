"""
Streamlit Dashboard for Stock Volatility Modeling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data.fetcher import fetch_stock_data
from data.preprocessing import calculate_returns, calculate_realized_volatility
from models.garch import GARCHModel, EGARCHModel, GJRGARCHModel
from models.ml_models import RandomForestVolatility, XGBoostVolatility
from models.stochastic_vol import HestonModel
from evaluation.metrics import calculate_all_metrics
from evaluation.backtesting import RollingWindowBacktest

# Page config
st.set_page_config(
    page_title="Stock Volatility Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Volatility Modeling Dashboard")
st.markdown("Compare different volatility forecasting models on stock data")

# Sidebar configuration
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Stock Ticker", value="SPY")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2020, 1, 1)
    )
with col2:
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime(2024, 1, 1)
    )

st.sidebar.subheader("Model Selection")
use_garch = st.sidebar.checkbox("GARCH(1,1)", value=True)
use_egarch = st.sidebar.checkbox("EGARCH(1,1)", value=True)
use_gjr = st.sidebar.checkbox("GJR-GARCH(1,1)", value=True)
use_rf = st.sidebar.checkbox("Random Forest", value=True)
use_xgb = st.sidebar.checkbox("XGBoost", value=True)
use_heston = st.sidebar.checkbox("Heston Model", value=False)

st.sidebar.subheader("Backtest Settings")
train_window = st.sidebar.slider("Training Window (days)", 100, 756, 504)
test_window = st.sidebar.slider("Test Window (days)", 5, 63, 21)


@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    """Load and preprocess data."""
    data = fetch_stock_data(ticker, str(start), str(end))
    returns = calculate_returns(data[['close']]).squeeze()
    return data, returns


@st.cache_data(ttl=3600)
def run_backtest(_models, returns, train_window, test_window):
    """Run backtest for all models."""
    backtest = RollingWindowBacktest(
        train_window=train_window,
        test_window=test_window,
        step_size=test_window
    )
    results = backtest.run(returns, _models, verbose=False)
    metrics = backtest.evaluate(results)
    return results, metrics


# Load data button
if st.sidebar.button("Load Data & Run Analysis", type="primary"):
    with st.spinner(f"Loading {ticker} data..."):
        try:
            data, returns = load_data(ticker, start_date, end_date)
            st.session_state['data'] = data
            st.session_state['returns'] = returns
            st.session_state['ticker'] = ticker
            st.success(f"Loaded {len(data)} days of {ticker} data")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

# Main content
if 'data' in st.session_state:
    data = st.session_state['data']
    returns = st.session_state['returns']
    ticker = st.session_state['ticker']

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "🔮 Model Fitting",
        "📈 Backtest Results",
        "📋 Metrics Comparison"
    ])

    # Tab 1: Data Overview
    with tab1:
        st.header(f"{ticker} Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(data))
        with col2:
            st.metric("Annualized Return", f"{returns.mean() * 252 * 100:.2f}%")
        with col3:
            st.metric("Annualized Volatility", f"{returns.std() * np.sqrt(252) * 100:.2f}%")
        with col4:
            st.metric("Sharpe Ratio", f"{(returns.mean() / returns.std()) * np.sqrt(252):.2f}")

        # Price chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=["Price", "Daily Returns", "Rolling Volatility (21-day)"]
        )

        fig.add_trace(
            go.Scatter(x=data.index, y=data['close'], name="Close Price", line=dict(color='blue')),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=returns.index, y=returns, name="Returns", marker_color='steelblue'),
            row=2, col=1
        )

        rolling_vol = returns.rolling(21).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol, name="Volatility", line=dict(color='red')),
            row=3, col=1
        )

        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Distribution
        st.subheader("Return Distribution")
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                returns, nbins=50,
                title="Return Histogram",
                labels={'value': 'Return', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max'],
                'Value': [
                    f"{returns.mean():.6f}",
                    f"{returns.std():.6f}",
                    f"{returns.skew():.4f}",
                    f"{returns.kurtosis():.4f}",
                    f"{returns.min():.6f}",
                    f"{returns.max():.6f}"
                ]
            })
            st.table(stats_df)

    # Tab 2: Model Fitting
    with tab2:
        st.header("Model Fitting Results")

        if st.button("Fit All Selected Models"):
            models_to_fit = {}

            if use_garch:
                models_to_fit['GARCH(1,1)'] = GARCHModel(p=1, q=1)
            if use_egarch:
                models_to_fit['EGARCH(1,1)'] = EGARCHModel(p=1, q=1)
            if use_gjr:
                models_to_fit['GJR-GARCH(1,1)'] = GJRGARCHModel(p=1, o=1, q=1)
            if use_rf:
                models_to_fit['Random Forest'] = RandomForestVolatility(n_estimators=50, lookback=20)
            if use_xgb:
                models_to_fit['XGBoost'] = XGBoostVolatility(n_estimators=50, lookback=20)
            if use_heston:
                models_to_fit['Heston'] = HestonModel()

            fitted_models = {}
            progress = st.progress(0)

            for i, (name, model) in enumerate(models_to_fit.items()):
                with st.spinner(f"Fitting {name}..."):
                    try:
                        model.fit(returns)
                        fitted_models[name] = model
                    except Exception as e:
                        st.warning(f"Failed to fit {name}: {e}")
                progress.progress((i + 1) / len(models_to_fit))

            st.session_state['fitted_models'] = fitted_models
            st.success(f"Fitted {len(fitted_models)} models successfully!")

        if 'fitted_models' in st.session_state:
            fitted_models = st.session_state['fitted_models']

            # Display conditional variance
            st.subheader("Conditional Variance Comparison")

            fig = go.Figure()
            squared_returns = returns ** 2
            fig.add_trace(go.Scatter(
                x=returns.index, y=squared_returns,
                name="Squared Returns",
                line=dict(color='gray', width=0.5),
                opacity=0.5
            ))

            colors = px.colors.qualitative.Set2
            for i, (name, model) in enumerate(fitted_models.items()):
                cond_var = model.get_conditional_variance()
                if cond_var is not None:
                    fig.add_trace(go.Scatter(
                        x=returns.index, y=cond_var,
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ))

            fig.update_layout(
                title="Conditional Variance vs Squared Returns",
                xaxis_title="Date",
                yaxis_title="Variance",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast
            st.subheader("Variance Forecasts")
            forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 10)

            forecast_df = pd.DataFrame({'Day': range(1, forecast_horizon + 1)})
            for name, model in fitted_models.items():
                try:
                    forecast = model.forecast_variance(forecast_horizon)
                    forecast_df[name] = forecast
                except Exception as e:
                    st.warning(f"Forecast failed for {name}: {e}")

            fig_forecast = px.line(
                forecast_df, x='Day', y=[c for c in forecast_df.columns if c != 'Day'],
                title=f"{forecast_horizon}-Day Variance Forecast"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

    # Tab 3: Backtest Results
    with tab3:
        st.header("Rolling Window Backtest")

        if st.button("Run Backtest"):
            models_dict = {}

            if use_garch:
                models_dict['GARCH(1,1)'] = GARCHModel(p=1, q=1)
            if use_egarch:
                models_dict['EGARCH(1,1)'] = EGARCHModel(p=1, q=1)
            if use_gjr:
                models_dict['GJR-GARCH(1,1)'] = GJRGARCHModel(p=1, o=1, q=1)
            if use_rf:
                models_dict['Random Forest'] = RandomForestVolatility(n_estimators=50, lookback=20)
            if use_xgb:
                models_dict['XGBoost'] = XGBoostVolatility(n_estimators=50, lookback=20)
            if use_heston:
                models_dict['Heston'] = HestonModel()

            with st.spinner("Running backtest (this may take a while)..."):
                try:
                    results, metrics = run_backtest(
                        models_dict, returns, train_window, test_window
                    )
                    st.session_state['backtest_results'] = results
                    st.session_state['backtest_metrics'] = metrics
                    st.success("Backtest completed!")
                except Exception as e:
                    st.error(f"Backtest failed: {e}")

        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']

            # Plot forecasts vs realized
            st.subheader("Forecasts vs Realized Variance")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=results.index, y=results['realized'],
                name="Realized",
                line=dict(color='black', width=2)
            ))

            model_cols = [c for c in results.columns if c != 'realized']
            colors = px.colors.qualitative.Set2
            for i, col in enumerate(model_cols):
                fig.add_trace(go.Scatter(
                    x=results.index, y=results[col],
                    name=col,
                    line=dict(color=colors[i % len(colors)]),
                    opacity=0.7
                ))

            fig.update_layout(
                title="Out-of-Sample Forecasts vs Realized Variance",
                xaxis_title="Date",
                yaxis_title="Variance",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cumulative loss
            st.subheader("Cumulative QLIKE Loss")

            realized = results['realized'].values
            cum_loss_df = pd.DataFrame({'Date': results.index})

            for col in model_cols:
                predicted = results[col].values
                pred_safe = np.maximum(predicted, 1e-10)
                real_safe = np.maximum(realized, 1e-10)
                qlike_loss = np.log(pred_safe) + real_safe / pred_safe
                cum_loss_df[col] = np.nancumsum(qlike_loss)

            fig_loss = px.line(
                cum_loss_df, x='Date', y=model_cols,
                title="Cumulative QLIKE Loss Over Time"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    # Tab 4: Metrics Comparison
    with tab4:
        st.header("Model Performance Metrics")

        if 'backtest_metrics' in st.session_state:
            metrics = st.session_state['backtest_metrics']

            # Display metrics table
            st.subheader("Evaluation Metrics")
            display_metrics = metrics[['rmse', 'mae', 'qlike', 'r_squared', 'mz_beta']].copy()
            display_metrics.columns = ['RMSE', 'MAE', 'QLIKE', 'R²', 'MZ Beta']
            st.dataframe(display_metrics.style.format("{:.6f}"), use_container_width=True)

            # Rankings
            st.subheader("Model Rankings")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**By RMSE (lower is better)**")
                ranking = metrics['rmse'].sort_values()
                for i, (model, val) in enumerate(ranking.items(), 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                    st.write(f"{emoji} {i}. {model}: {val:.6f}")

            with col2:
                st.markdown("**By MAE (lower is better)**")
                ranking = metrics['mae'].sort_values()
                for i, (model, val) in enumerate(ranking.items(), 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                    st.write(f"{emoji} {i}. {model}: {val:.6f}")

            with col3:
                st.markdown("**By QLIKE (lower is better)**")
                ranking = metrics['qlike'].sort_values()
                for i, (model, val) in enumerate(ranking.items(), 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                    st.write(f"{emoji} {i}. {model}: {val:.6f}")

            # Bar chart comparison
            st.subheader("Visual Comparison")

            metric_to_plot = st.selectbox(
                "Select metric to visualize",
                options=['rmse', 'mae', 'qlike', 'r_squared']
            )

            fig_bar = px.bar(
                x=metrics.index,
                y=metrics[metric_to_plot],
                title=f"{metric_to_plot.upper()} by Model",
                labels={'x': 'Model', 'y': metric_to_plot.upper()},
                color=metrics[metric_to_plot],
                color_continuous_scale='RdYlGn_r' if metric_to_plot != 'r_squared' else 'RdYlGn'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Mincer-Zarnowitz interpretation
            st.subheader("Mincer-Zarnowitz Regression Interpretation")
            st.markdown("""
            The Mincer-Zarnowitz regression tests forecast efficiency:
            - **α (alpha)**: Should be close to 0 for unbiased forecasts
            - **β (beta)**: Should be close to 1 for efficient forecasts
            - Values significantly different from 0 and 1 indicate forecast issues
            """)

            mz_df = metrics[['mz_alpha', 'mz_beta', 'mz_r_squared']].copy()
            mz_df.columns = ['Alpha', 'Beta', 'R²']
            st.dataframe(mz_df.style.format("{:.4f}"), use_container_width=True)

        else:
            st.info("Run the backtest in the 'Backtest Results' tab first to see metrics.")

else:
    st.info("👈 Configure settings in the sidebar and click 'Load Data & Run Analysis' to begin.")

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit | "
    "Models: GARCH, EGARCH, GJR-GARCH, Random Forest, XGBoost, Heston"
)
