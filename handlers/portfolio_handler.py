
import pandas as pd
from pydantic import ValidationError

from models.asset import Asset  # Assuming you have an Asset class defined as previously discussed
from models.portfolio import Portfolio  # Assuming you have a Portfolio class that manages a collection of Assets
from models.activity import Activity
from data.db_manager import DatabaseManager


class PortfolioHandler:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.db_manager = DatabaseManager()
        self.db_manager.initialize_db()
        self.portfolios = {}  # Dictionary to hold portfolio names and their corresponding Portfolio objects

        self.load_actions_from_excel()

    def load_actions_from_excel(self):
        """
        Loads actions from the Excel file and processes each action.
        """
        activity_df = pd.read_excel(self.excel_path)
        
        for _, row in activity_df.iterrows():
            # Convert dataframe row to pyndantic model
            try:
                # Convert the row to a dictionary and then unpack into a User instance
                activity = Activity(**row.to_dict())
            except ValidationError as e:
                # Handle cases where the data doesn't match the model
                print(f"Error converting row to Activity: {e}")
            
            # Process the action
            self.process_activity(activity)

    def process_activity(self, activity):
        """
        Processes a single action by updating or creating portfolios and assets accordingly.
        """
        # portfolio_name = activity['PortfolioName']
        # asset_name = activity['AssetName']
        # action_type = activity['ActionType']
        # date = activity['Date']
        # amount = activity['Amount']
        # currency = activity['Currency']
        # remarks = activity.get('Remarks', '')

        # Retrieve or create the Portfolio object
        portfolio = self.portfolios.get(activity.Account)
        if portfolio is None:
            portfolio = Portfolio(activity.Account)
            self.portfolios[activity.Account] = portfolio

        # Retrieve or create the Asset object
        asset = portfolio.get_asset(activity.Asset)
        if asset is None:
            asset = Asset(activity.Asset, ticker_symbol='')  # Assuming ticker_symbol or similar identifier is available
            portfolio.add_asset(asset)

        # Depending on action type, update the asset (and portfolio if necessary)
        # if action_type == 'Transaction':
        #     asset.add_transaction(Act(date=date, amount=amount, currency=currency, remarks=remarks))
        # Add more conditions for different action types like 'Dividend', 'PriceChange', etc.

        # Optionally, update portfolio-level metrics or balances if necessary


