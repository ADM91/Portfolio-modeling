
import streamlit as st
from services.portfolio_service import PortfolioService
from config import settings

def main():
    st.title("Multi-Currency Asset Tracker")
    
    # Use a default path or get from environment
    excel_path = st.sidebar.text_input(
        "Excel File Path", 
        value="data/portfolio.xlsx",
        help="Path to your portfolio Excel file"
    )

    portfolio_service = PortfolioService(excel_path)

    menu = ["Dashboard", "Add Activity", "Portfolio Analysis", "Data Visualization"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Dashboard":
        show_dashboard(portfolio_service)
    elif choice == "Add Activity":
        add_activity(portfolio_service)
    elif choice == "Portfolio Analysis":
        show_portfolio_analysis(portfolio_service)
    elif choice == "Data Visualization":
        show_data_visualization(portfolio_service)

def show_dashboard(portfolio_service):
    # Display summary of all portfolios
    pass

def add_activity(portfolio_service):
    # Form for adding new activities
    pass

def show_portfolio_analysis(portfolio_service):
    # Display detailed analysis of selected portfolio
    pass

def show_data_visualization(portfolio_service):
    # Charts and graphs of portfolio performance
    pass

if __name__ == "__main__":
    main()