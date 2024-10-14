
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
