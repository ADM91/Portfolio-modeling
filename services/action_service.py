
import logging
import pandas as pd
from typing import List

from database.access import DatabaseAccess
from database.entities import Action


class ActionService:
    """
    Service class for managing actions - reads from input, validates, and inserts into database
    """

    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

    def read_actions_from_excel_to_action_list(self, excel_path: str) -> list[Action]:
        """
        Loads actions from excel, interprets and validates them
        """
        action_list = []

        try:
            # Read actions from excel
            action_df = pd.read_excel(excel_path)
        except Exception as e:
            logging.error(f"Failed to read Excel file: {e}")
            return action_list
        
        for _, row in action_df.iterrows():
            try:
                # Convert date string to datetime object
                row['date'] =  pd.to_datetime(row['Date'], dayfirst=True, format='mixed', errors='coerce')
                row['date'] = pd.to_datetime(row['date'].date())

                # Get portfolio_id based on portfolio name
                portfolio = self.db_access.get_portfolio_by_name(row['Portfolio'])
                if portfolio:
                    row['portfolio_id'] = portfolio.id
                else:
                    raise ValueError(f"Invalid portfolio name: {row['portfolio_name']}")

                # Get action_type_id based on action_type name
                action_type = self.db_access.get_action_type_by_name(row['Action'])
                if action_type:
                    row['action_type_id'] = action_type.id
                else:
                    raise ValueError(f"Invalid action type: {row['action_type']}")

                # Get asset_id based on asset symbol
                asset = self.db_access.get_asset_by_code(row['Asset'])
                if asset:
                    row['asset_id'] = asset.id
                else:
                    raise ValueError(f"Invalid asset symbol: {row['Asset']}")

                # Get currency_id based on currency symbol
                currency = self.db_access.get_asset_by_code(row['Currency'])
                if currency:
                    row['currency_id'] = currency.id
                else:
                    raise ValueError(f"Invalid currency symbol: {row['Currency']}")

                # Remove unnecessary columns
                row = row.drop(['Asset', 'Currency', 'Action', 'Portfolio'])

                # add is_processed
                row['is_processed'] = False

                # rename columns to match database
                row = row.rename({'Date': 'date', 'Price': 'price', 'Quantity': 'quantity', 'Fee': 'fee', 'Platform': 'platform', 'Comment': 'comment'})

                # Convert the row to a dictionary and then create an Action instance
                action_list.append(row.to_dict())

            except Exception as e:
                logging.error(f"Error converting row to Action: {e}")
                logging.error(f"action: {row}")
        
        return action_list

    def insert_actions(self, actions_data: List[dict]):
        """
        Inserts an action into the database if it doesn't already exist.
        """
        self.db_access.insert_if_not_exists(Action, actions_data)

    def update_action(self):
        pass

    def delete_action(self):
        pass


if __name__ == "__main__":

    AS = ActionService(DatabaseAccess())
    actions = AS.read_actions_from_excel_to_action_list("_junk/test_action_data.xlsx")
    AS.insert_actions(actions)

    print('done')