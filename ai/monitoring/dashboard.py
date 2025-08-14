# ai/monitoring/dashboard.py
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


class TradingDashboard:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        st.set_page_config(
            page_title="RNN Trading System",
            page_icon="🤖",
            layout="wide"
        )

    def render_main_dashboard(self):
        st.title("🤖 RNN Trading System - Live Performance")

        # Key performance metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Portfolio Value",
                f"${self.get_portfolio_value():,.2f}",
                f"{self.get_daily_pnl():+.2f}%"
            )

        with col2:
            st.metric(
                "Total Return",
                f"{self.get_total_return():+.2f}%",
                f"vs S&P500: {self.get_alpha():+.2f}%"
            )

        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{self.get_sharpe_ratio():.2f}",
                f"{self.get_sharpe_change():+.2f}"
            )

        with col4:
            st.metric(
                "Max Drawdown",
                f"{self.get_max_drawdown():.2f}%",
                f"{self.get_drawdown_change():+.2f}%"
            )

        with col5:
            st.metric(
                "Win Rate",
                f"{self.get_win_rate():.1f}%",
                f"{self.get_win_rate_change():+.1f}%"
            )

        # Performance charts
        self.render_equity_curve()
        self.render_position_analysis()
        self.render_signal_analysis()

    def render_equity_curve(self):
        """Real-time equity curve with benchmark comparison."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Portfolio Value vs Benchmark', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )

        # Portfolio equity curve
        portfolio_data = self.get_portfolio_history()
        benchmark_data = self.get_benchmark_history()

        fig.add_trace(
            go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data['cumulative_return'],
                name='RNN Strategy',
                line={"color": '#00ff00', "width": 2}
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data['cumulative_return'],
                name='S&P 500',
                line={"color": '#1f77b4', "width": 1}
            ),
            row=1, col=1
        )

        # Drawdown chart
        drawdown = self.calculate_drawdown(portfolio_data['cumulative_return'])
        fig.add_trace(
            go.Scatter(
                x=portfolio_data.index,
                y=drawdown,
                fill='tonexty',
                name='Drawdown',
                line={"color": 'red'}
            ),
            row=2, col=1
        )

        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
