
import streamlit as st
from services.portfolio_service import PortfolioHandler
from config import settings

def main():
    st.title("Multi-Currency Asset Tracker")
    
    # Use a default path or get from environment
    excel_path = st.sidebar.text_input(
        "Excel File Path", 
        value="data/portfolio.xlsx",
        help="Path to your portfolio Excel file"
    )
    
    portfolio_handler = PortfolioHandler(excel_path)
    
    menu = ["Dashboard", "Add Activity", "Portfolio Analysis", "Data Visualization"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Dashboard":
        show_dashboard(portfolio_handler)
    elif choice == "Add Activity":
        add_activity(portfolio_handler)
    elif choice == "Portfolio Analysis":
        show_portfolio_analysis(portfolio_handler)
    elif choice == "Data Visualization":
        show_data_visualization(portfolio_handler)

def show_dashboard(portfolio_handler):
    # Display summary of all portfolios
    pass

def add_activity(portfolio_handler):
    # Form for adding new activities
    pass

def show_portfolio_analysis(portfolio_handler):
    # Display detailed analysis of selected portfolio
    pass

def show_data_visualization(portfolio_handler):
    # Charts and graphs of portfolio performance
    pass

if __name__ == "__main__":
    main()