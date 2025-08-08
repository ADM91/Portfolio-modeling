"""
Portfolio Dashboard - Streamlit Application
Interactive dashboard for portfolio analysis with multi-currency support
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import numpy as np

from api_client import PortfolioAPIClient

# Page configuration
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API client
@st.cache_resource
def get_api_client():
    return PortfolioAPIClient()

api_client = get_api_client()

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stSelectbox > div > div > select {
        background-color: #ffffff;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Portfolio Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Base Asset Selection (Currency)
        st.subheader("💱 Base Asset for Calculations")
        available_assets = api_client.get_available_assets()
        
        if not available_assets:
            st.error("❌ No assets found in database. Please ensure your database is populated with data.")
            st.stop()
        
        asset_options = {}
        for asset in available_assets:
            label = f"{asset['symbol']} - {asset['name']} ({asset['type']})"
            asset_options[label] = asset['id']
        
        selected_base_asset_label = st.selectbox(
            "Select base asset:",
            options=list(asset_options.keys()),
            index=0,
            help="All metrics will be calculated in terms of this asset"
        )
        base_currency_id = asset_options[selected_base_asset_label]
        base_symbol = selected_base_asset_label.split(' - ')[0]
        
        st.info(f"📈 All values shown in **{base_symbol}** terms")
        
        # Portfolio Selection
        st.subheader("📁 Portfolio Selection")
        available_portfolios = api_client.get_available_portfolios()
        
        portfolio_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in available_portfolios}
        selected_portfolio_labels = st.multiselect(
            "Select portfolios:",
            options=list(portfolio_options.keys()),
            default=list(portfolio_options.keys())[:2],  # Select first 2 by default
            help="Select one or more portfolios to analyze"
        )
        selected_portfolio_ids = [portfolio_options[label] for label in selected_portfolio_labels]
        
        # Asset Selection
        st.subheader("🏷️ Asset Selection")
        if selected_portfolio_ids:
            available_portfolio_assets = api_client.get_available_portfolio_assets(selected_portfolio_ids)
            
            asset_options = {f"{a['symbol']} - {a['name']}": a['id'] for a in available_portfolio_assets}
            selected_asset_labels = st.multiselect(
                "Select assets:",
                options=list(asset_options.keys()),
                default=list(asset_options.keys()),  # Select all by default
                help="Select specific assets or leave empty for all assets"
            )
            selected_asset_ids = [asset_options[label] for label in selected_asset_labels]
        else:
            st.warning("Please select at least one portfolio first")
            selected_asset_ids = []
        
        # Time Range Selection
        st.subheader("📅 Time Range")
        time_option = st.radio(
            "Select time range:",
            options=["Preset", "Custom"],
            index=0
        )
        
        if time_option == "Preset":
            timeframe = st.selectbox(
                "Timeframe:",
                options=["30d", "90d", "ytd", "1y", "max"],
                index=2,  # Default to YTD
                help="Preset time ranges"
            )
            start_date = None
            end_date = None
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date:",
                    value=date.today() - timedelta(days=365),
                    max_value=date.today()
                )
            with col2:
                end_date = st.date_input(
                    "End Date:",
                    value=date.today(),
                    max_value=date.today()
                )
            timeframe = "custom"
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content area
    if not selected_portfolio_ids or not selected_asset_ids:
        st.warning("⚠️ Please select at least one portfolio and one asset to view data")
        return
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            # Get data from API
            if time_option == "Custom" and start_date and end_date:
                holdings_df = api_client.get_holdings_value(
                    base_currency_id, selected_portfolio_ids, selected_asset_ids,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                invested_df = api_client.get_value_invested(
                    base_currency_id, selected_portfolio_ids, selected_asset_ids,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                returns_df = api_client.get_time_weighted_return(
                    base_currency_id, selected_portfolio_ids, selected_asset_ids,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
            else:
                holdings_df = api_client.get_holdings_value(
                    base_currency_id, selected_portfolio_ids, selected_asset_ids, timeframe
                )
                invested_df = api_client.get_value_invested(
                    base_currency_id, selected_portfolio_ids, selected_asset_ids, timeframe
                )
                returns_df = api_client.get_time_weighted_return(
                    base_currency_id, selected_portfolio_ids, selected_asset_ids, timeframe
                )
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    # Check if we have data
    if holdings_df.empty and invested_df.empty and returns_df.empty:
        st.warning("📭 No data available for the selected criteria")
        return
    
    # Summary metrics
    display_summary_metrics(holdings_df, invested_df, returns_df, base_symbol)
    
    # Charts
    st.header("📈 Portfolio Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["💰 Holdings Value", "💸 Investment Flow", "📊 Performance"])
    
    with tab1:
        display_holdings_chart(holdings_df, base_symbol)
    
    with tab2:
        display_investment_flow_chart(holdings_df, invested_df, base_symbol)
    
    with tab3:
        display_performance_chart(returns_df, base_symbol)

def display_summary_metrics(holdings_df: pd.DataFrame, invested_df: pd.DataFrame, 
                          returns_df: pd.DataFrame, base_symbol: str):
    """Display summary metric cards"""
    st.header("📋 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not holdings_df.empty:
            current_value = holdings_df.groupby('date')['holding_value'].sum().iloc[-1]
            st.metric(
                label=f"Current Portfolio Value",
                value=f"{current_value:,.2f} {base_symbol}",
                help="Total current value of all selected holdings"
            )
        else:
            st.metric("Current Portfolio Value", "No data")
    
    with col2:
        if not invested_df.empty:
            total_invested = invested_df.groupby('date')['cash_flow_cumulative'].sum().iloc[-1]
            st.metric(
                label=f"Total Invested",
                value=f"{total_invested:,.2f} {base_symbol}",
                help="Total amount invested (cumulative cash flows)"
            )
        else:
            st.metric("Total Invested", "No data")
    
    with col3:
        if not holdings_df.empty and not invested_df.empty:
            current_value = holdings_df.groupby('date')['holding_value'].sum().iloc[-1]
            total_invested = invested_df.groupby('date')['cash_flow_cumulative'].sum().iloc[-1]
            unrealized_pnl = current_value - total_invested
            pnl_pct = (unrealized_pnl / total_invested * 100) if total_invested != 0 else 0
            
            st.metric(
                label=f"Unrealized P&L",
                value=f"{unrealized_pnl:,.2f} {base_symbol}",
                delta=f"{pnl_pct:.2f}%",
                help="Difference between current value and total invested"
            )
        else:
            st.metric("Unrealized P&L", "No data")
    
    with col4:
        if not returns_df.empty:
            total_return = (returns_df['time_weighted_return'].iloc[-1] - 1) * 100
            st.metric(
                label="Total Return",
                value=f"{total_return:.2f}%",
                help="Time-weighted total return"
            )
        else:
            st.metric("Total Return", "No data")

def display_holdings_chart(holdings_df: pd.DataFrame, base_symbol: str):
    """Display holdings value chart"""
    if holdings_df.empty:
        st.warning("No holdings data available")
        return
    
    st.subheader(f"Portfolio Holdings Value Over Time ({base_symbol})")
    
    # Aggregate by date
    daily_holdings = holdings_df.groupby('date')['holding_value'].sum().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_holdings['date'],
        y=daily_holdings['holding_value'],
        mode='lines',
        name='Holdings Value',
        line=dict(color='#1f77b4', width=2),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    fig.update_layout(
        title=f"Portfolio Holdings Value ({base_symbol})",
        xaxis_title="Date",
        yaxis_title=f"Value ({base_symbol})",
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show breakdown by portfolio/asset if multiple
    if len(holdings_df['portfolio_id'].unique()) > 1 or len(holdings_df['asset_id'].unique()) > 1:
        with st.expander("📊 Detailed Breakdown"):
            st.dataframe(
                holdings_df.pivot_table(
                    index='date', 
                    columns=['portfolio_id', 'asset_id'], 
                    values='holding_value', 
                    fill_value=0
                ).tail(10),
                use_container_width=True
            )

def display_investment_flow_chart(holdings_df: pd.DataFrame, invested_df: pd.DataFrame, base_symbol: str):
    """Display investment flow comparison chart"""
    if holdings_df.empty or invested_df.empty:
        st.warning("Insufficient data for investment flow analysis")
        return
    
    st.subheader(f"Holdings vs Investment Flow ({base_symbol})")
    
    # Aggregate data
    daily_holdings = holdings_df.groupby('date')['holding_value'].sum().reset_index()
    daily_invested = invested_df.groupby('date')['cash_flow_cumulative'].sum().reset_index()
    
    # Merge data
    merged_df = pd.merge(daily_holdings, daily_invested, on='date', how='outer').ffill()
    
    fig = go.Figure()
    
    # Holdings value
    fig.add_trace(go.Scatter(
        x=merged_df['date'],
        y=merged_df['holding_value'],
        mode='lines',
        name='Current Holdings Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Invested value
    fig.add_trace(go.Scatter(
        x=merged_df['date'],
        y=merged_df['cash_flow_cumulative'],
        mode='lines',
        name='Cumulative Invested',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title=f"Holdings Value vs Investment Flow ({base_symbol})",
        xaxis_title="Date",
        yaxis_title=f"Value ({base_symbol})",
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_performance_chart(returns_df: pd.DataFrame, base_symbol: str):
    """Display performance returns chart"""
    if returns_df.empty:
        st.warning("No performance data available")
        return
    
    st.subheader(f"Time-Weighted Returns (Base: {base_symbol})")
    
    # Convert to percentage
    returns_df_pct = returns_df.copy()
    returns_df_pct['time_weighted_return_pct'] = (returns_df_pct['time_weighted_return'] - 1) * 100
    returns_df_pct['daily_return_pct'] = returns_df_pct['daily_adjusted_return'] * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative Returns (%)', 'Daily Returns (%)'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Cumulative returns
    fig.add_trace(
        go.Scatter(
            x=returns_df_pct['date'],
            y=returns_df_pct['time_weighted_return_pct'],
            mode='lines',
            name='Cumulative Return',
            line=dict(color='#2ca02c', width=2)
        ),
        row=1, col=1
    )
    
    # Daily returns
    colors = ['red' if x < 0 else 'green' for x in returns_df_pct['daily_return_pct']]
    fig.add_trace(
        go.Bar(
            x=returns_df_pct['date'],
            y=returns_df_pct['daily_return_pct'],
            name='Daily Return',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"Portfolio Performance Analysis (Base: {base_symbol})",
        showlegend=True,
        height=600,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance statistics
    if len(returns_df) > 1:
        with st.expander("📈 Performance Statistics"):
            col1, col2, col3 = st.columns(3)
            
            daily_returns = returns_df['daily_adjusted_return'].dropna()
            
            with col1:
                st.metric("Total Return", f"{(returns_df['time_weighted_return'].iloc[-1] - 1) * 100:.2f}%")
                st.metric("Volatility (Annualized)", f"{daily_returns.std() * np.sqrt(365) * 100:.2f}%")
            
            with col2:
                st.metric("Best Day", f"{daily_returns.max() * 100:.2f}%")
                st.metric("Worst Day", f"{daily_returns.min() * 100:.2f}%")
            
            with col3:
                positive_days = (daily_returns > 0).sum()
                total_days = len(daily_returns)
                win_rate = positive_days / total_days * 100 if total_days > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.metric("Total Trading Days", f"{total_days}")

if __name__ == "__main__":
    main()
