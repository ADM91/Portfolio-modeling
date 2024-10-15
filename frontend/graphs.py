
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns


def generate_color_palette(n):
    return plt.cm.get_cmap('tab20')(np.linspace(0, 1, n))


def multi_portfolio_value_graph(df_melted):  

    # Assuming df_melted is your new melted dataframe
    # It should have columns: 'date', 'portfolio_id', 'asset_id', 'value'

    # Convert date to datetime if it's not already
    df_melted['date'] = pd.to_datetime(df_melted['date'])

    # Sort the dataframe
    df_melted = df_melted.sort_values(['date', 'portfolio_id', 'asset_id'])

    # Get unique portfolio_ids
    portfolios = df_melted['portfolio_id'].unique()

    # Sort portfolios to ensure consistent stacking order
    sorted_portfolios = sorted(portfolios)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(df_melted['date'].unique()))

    for portfolio in sorted_portfolios:
        portfolio_data = df_melted[df_melted['portfolio_id'] == portfolio]
        unique_assets = portfolio_data['asset_id'].unique()
        
        # Generate a color palette for assets in this portfolio
        asset_colors = generate_color_palette(len(unique_assets))
        
        for i, asset in enumerate(unique_assets):
            asset_data = portfolio_data[portfolio_data['asset_id'] == asset]
            values = asset_data.set_index('date')['value_holdings'].reindex(df_melted['date'].unique()).fillna(0).values
            
            ax.fill_between(df_melted['date'].unique(), bottom, bottom + values, 
                            label=f'Portfolio {portfolio} - Asset {asset}',
                            color=asset_colors[i], alpha=0.7)
            
            bottom += values

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Stacked Portfolio Assets Over Time')

    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('frontend/graphs/multi_portfolio_value.png')


def holdings_and_invested_value_graph(result_holdings_value, result_invested_value):
    
    # Merge the dataframes on date
    merged_df = pd.merge(result_holdings_value, result_invested_value, on=['date','portfolio_id','asset_id'], how='left').sort_values('date')
    
    # Forward fill any missing values
    merged_df[['value_holdings', 'value_invested_cumulative']] = merged_df[['value_holdings', 'value_invested_cumulative']].ffill()

    merged_df_agg = merged_df.groupby('date').sum().reset_index()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert values to millions of ISK
    holdings = merged_df_agg['value_holdings']
    invested = merged_df_agg['value_invested_cumulative']
    dates = merged_df_agg['date']

    # Plot holdings in base currency
    ax.plot(dates, holdings, color='blue', label='Holdings in Base Currency')

    # Create shaded area for value invested
    ax.fill_between(dates, 0, invested, 
                    where=(invested >= 0), color='red', alpha=0.3, 
                    label='Positive Value Invested')
    ax.fill_between(dates, 0, invested, 
                    where=(invested < 0), color='green', alpha=0.3, 
                    label='Negative Value Invested')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('currency')
    ax.set_title('Holdings in Base Currency and Value Invested Over Time')

    # Add legend
    ax.legend(loc='upper left')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('frontend/graphs/holdings_and_invested_value.png')
    print('Plot saved as holdings_and_invested_value.png')


def holdings_and_invested_value_graph_v2(result_holdings_value, result_invested_value):
    # Merge the dataframes on date
    merged_df = pd.merge(result_holdings_value, result_invested_value, on=['date','portfolio_id','asset_id'], how='outer').sort_values('date')
    
    # Forward fill any missing values
    merged_df[['value_holdings', 'value_invested_cumulative']] = merged_df[['value_holdings', 'value_invested_cumulative']].ffill()

    merged_df_agg = merged_df.groupby('date').sum().reset_index()

    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Convert values to millions of ISK
    holdings = merged_df_agg['value_holdings']
    invested = merged_df_agg['value_invested_cumulative']
    dates = merged_df_agg['date']

    # Plot holdings in base currency
    ax1.plot(dates, holdings, color='blue', label='Holdings in Base Currency')

    # Create shaded area for value invested
    ax1.fill_between(dates, 0, invested, 
                    where=(invested >= 0), color='red', alpha=0.3, 
                    label='Positive Value Invested')
    ax1.fill_between(dates, 0, invested, 
                    where=(invested < 0), color='green', alpha=0.3, 
                    label='Negative Value Invested')

    # Set labels and title for the first subplot
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Currency')
    ax1.set_title('Holdings in Base Currency and Value Invested Over Time')
    ax1.legend(loc='upper left')

    # Prepare data for monthly value invested bar chart
    monthly_invested = result_invested_value.groupby(result_invested_value['date'].dt.to_period('M'))['value_invested_discrete'].sum().reset_index()
    monthly_invested['date'] = monthly_invested['date'].dt.to_timestamp()

    # Plot monthly value invested bar chart
    ax2.bar(monthly_invested['date'], monthly_invested['value_invested_discrete'], 
            width=25, align='center', color='purple', alpha=0.7)

    # Set labels and title for the second subplot
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Monthly Value Invested')
    ax2.set_title('Monthly Value Invested')

    # Format x-axis to show dates properly
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('frontend/graphs/holdings_and_invested_value_graph_v2.png')


def holdings_and_cost_basis_graph(holdings_df: pd.DataFrame, cost_basis_df: pd.DataFrame, 
                                 portfolio_id: int, asset_id: int, currency_id: int) -> None:
    """
    Plot the value of holdings and cost basis for a specific asset in a portfolio.

    Args:
        holdings_df (pd.DataFrame): DataFrame output from get_holdings_in_base_currency_general
        cost_basis_df (pd.DataFrame): DataFrame output from get_cost_basis
        portfolio_id (int): The ID of the portfolio to plot
        asset_id (int): The ID of the asset to plot
        currency_id (int): The ID of the currency used

    Returns:
        None (displays the plot)
    """
    # Filter holdings data for the specific portfolio and asset
    asset_holdings = holdings_df[(holdings_df['portfolio_id'] == portfolio_id) & 
                                 (holdings_df['asset_id'] == asset_id)]

    # Merge holdings and cost basis data
    merged_df = pd.merge(asset_holdings, cost_basis_df, on='date', how='outer')
    merged_df = merged_df.sort_values('date')

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot holdings value
    ax.plot(merged_df['date'], merged_df['value_holdings'], label='Holdings Market Value', color='blue')

    # Plot total cost basis
    ax.plot(merged_df['date'], merged_df['cumulative_cost'], label='Cost Basis Value', color='red')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Value (Currency ID: {currency_id})')
    ax.set_title(f'Holdings Value vs Cost Basis for Portfolio {portfolio_id}, Asset {asset_id}')

    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

    # Add legend
    ax.legend()

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.savefig('frontend/graphs/holdings_and_cost_basis_graph.png')


def price_and_cost_basis(holdings_df: pd.DataFrame, cost_basis_df: pd.DataFrame, 
                                    portfolio_id: int, asset_id: int, currency_id: int) -> None:
    """
    Plot the asset market value and cost basis for a specific asset in a portfolio.

    Args:
        holdings_df (pd.DataFrame): DataFrame containing asset holdings data
        cost_basis_df (pd.DataFrame): DataFrame containing cost basis data
        portfolio_id (int): The ID of the portfolio to plot
        asset_id (int): The ID of the asset to plot
        currency_id (int): The ID of the currency used

    Returns:
        None (displays the plot)
    """
    # Filter holdings data for the specific portfolio and asset
    asset_holdings = holdings_df[(holdings_df['portfolio_id'] == portfolio_id) & 
                                 (holdings_df['asset_id'] == asset_id)]

    # Merge holdings and cost basis data
    merged_df = pd.merge(asset_holdings, cost_basis_df, on='date', how='outer')
    merged_df = merged_df.sort_values('date')
    merged_df['market_price'] = merged_df['value_holdings'] / merged_df['cumulative_quantity']


    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot asset market value
    ax.plot(merged_df['date'], merged_df['market_price'], label='Market Price', color='green')

    # Plot cost basis
    ax.plot(merged_df['date'], merged_df['cost_basis'], label='Cost Basis Price', color='orange')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Value (Currency ID: {currency_id})')
    ax.set_title(f'Asset Market Price vs Cost Basis for Portfolio {portfolio_id}, Asset {asset_id}')

    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

    # Add legend
    ax.legend()

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.savefig('frontend/graphs/price_and_cost_basis.png')


def unrealized_gain_loss_percentage_graph(unrealized_gain_loss_df: pd.DataFrame, 
                                          portfolio_id: int, asset_id: int, currency_id: int) -> None:
    """
    Plot the unrealized gain/loss percentage for a specific asset in a portfolio.

    Args:
        unrealized_gain_loss_df (pd.DataFrame): DataFrame output from get_unrealized_gain_loss
        portfolio_id (int): The ID of the portfolio to plot
        asset_id (int): The ID of the asset to plot
        currency_id (int): The ID of the currency used

    Returns:
        None (displays the plot)
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot unrealized gain/loss percentage
    ax.plot(unrealized_gain_loss_df['date'], unrealized_gain_loss_df['unrealized_gain_loss_percentage'], 
            label='Unrealized Gain/Loss %', color='green')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Unrealized Gain/Loss (%)')
    ax.set_title(f'Unrealized Gain/Loss Percentage for Portfolio {portfolio_id}, Asset {asset_id}, Currency {currency_id}')

    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

    # Add legend
    ax.legend()

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add a horizontal line at y=0 to show the break-even point
    ax.axhline(y=0, color='r', linestyle='--')

    # Show the plot
    plt.tight_layout()
    plt.savefig('frontend/graphs/unrealized_gain_loss_percentage_graph.png')



def total_unrealized_gain_loss_graph(unrealized_gain_loss_df):
    # Aggregate data by date
    df_agg = unrealized_gain_loss_df.groupby('date').agg({
        'cumulative_cost': 'sum',
        'value_in_currency': 'sum',
        'unrealized_gain_loss': 'sum'
    }).reset_index()

    df_agg['unrealized_gain_loss_percentage'] = (df_agg['unrealized_gain_loss'] / df_agg['cumulative_cost']) * 100

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    dates = df_agg['date']

    # Plot cumulative cost and value in currency
    ax.plot(dates, df_agg['unrealized_gain_loss_percentage'], color='blue', label='Cumulative Cost')
    # Add a horizontal line at y=0 to show the break-even point
    ax.axhline(y=0, color='r', linestyle='--')

    # Set labels and title for the first subplot
    ax.set_xlabel('Date')
    ax.set_ylabel('Currency')
    ax.set_title('Cumulative Cost and Value in Currency Over Time')
    ax.legend(loc='upper left')

    # # Plot unrealized gain/loss
    # ax2.bar(dates, df_agg['unreali_gain_loss_percentage'], 
    #         width=25, align='center'zed, color='purple', alpha=0.7)

    # Set labels and title for the second subplot
    # ax2.set_xlabel('Date')
    # ax2.set_ylabel('Unrealized Gain/Loss')
    # ax2.set_title('Unrealized Gain/Loss Over Time')

    # Format x-axis to show dates properly
    # ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('frontend/graphs/total_unrealized_gain_loss_percentage_graph.png')