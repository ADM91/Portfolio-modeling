# action_service.py

import logging
import pandas as pd
from typing import List, Dict
from sqlalchemy.orm import Session
from datetime import datetime
import time

from database.access import with_session, DatabaseAccess
from database.entities import Action


class ActionService:
    """
    Class for managing actions related to portfolios.
    """
    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

    @with_session
    def read_actions_from_excel(self, session: Session, excel_path: str) -> List[Dict]:
        """
        Loads actions from excel, interprets and validates them.
        
        :param excel_path: Path to the Excel file
        :return: List of dictionaries containing action data
        """
        action_list = []

        try:
            action_df = pd.read_excel(excel_path)
        except Exception as e:
            logging.error(f"Failed to read Excel file: {e}")
            return action_list
        
        for _, row in action_df.iterrows():
            try:
                row['date'] = pd.to_datetime(row['Date'], dayfirst=True, format='mixed', errors='coerce')
                row['date'] = pd.to_datetime(row['date'].date())

                portfolio = self.db_access.get_portfolio_by_name(session, row['Portfolio'])
                if portfolio:
                    row['portfolio_id'] = portfolio.id
                else:
                    raise ValueError(f"Invalid portfolio name: {row['Portfolio']}")

                action_type = self.db_access.get_action_type_by_name(session, row['Action'])
                if action_type:
                    row['action_type_id'] = action_type.id
                else:
                    raise ValueError(f"Invalid action type: {row['Action']}")

                asset = self.db_access.get_asset_by_code(session, row['Asset'])
                if asset:
                    row['asset_id'] = asset.id
                else:
                    raise ValueError(f"Invalid asset symbol: {row['Asset']}")

                currency = self.db_access.get_asset_by_code(session, row['Currency'])
                if currency:
                    row['currency_id'] = currency.id
                else:
                    raise ValueError(f"Invalid currency symbol: {row['Currency']}")

                row = row.drop(['Asset', 'Currency', 'Action', 'Portfolio'])
                row['is_processed'] = False
                row = row.rename({'Date': 'date', 'Price': 'price', 'Quantity': 'quantity', 'Fee': 'fee', 'Platform': 'platform', 'Comment': 'comment'})
                row = row.where((pd.notnull(row)), None)

                action_list.append(row.to_dict())

            except Exception as e:
                logging.error(f"Error converting row to Action: {e}")
                logging.error(f"action: {row}")
        
        return action_list

    @with_session
    def insert_actions(self, session: Session, actions_data: List[Dict]) -> None:
        """
        Inserts actions into the database if they don't already exist.
        
        :param actions_data: List of dictionaries containing action data
        """
        filter_fields = ['portfolio_id', 'action_type_id', 'date', 'asset_id', 'currency_id', 'price', 'quantity']
        self.db_access.insert_if_not_exists(session, Action, actions_data, filter_fields)

    @with_session
    def update_action(self, session: Session, action_id: int, updated_data: Dict) -> None:
        """
        Updates an existing action in the database.
        
        :param action_id: ID of the action to update
        :param updated_data: Dictionary containing updated action data
        """
        # Implementation for updating an action
        pass

    @with_session
    def delete_action(self, session: Session, action_id: int) -> None:
        """
        Deletes an action from the database.
        
        :param action_id: ID of the action to delete
        """
        # Implementation for deleting an action
        pass

    @with_session
    def process_actions(self, session: Session):

        # Get unprocessed actions
        unprocessed_actions = self.db_access.get_unprocessed_actions(session)

        # Process unprocessed actions, update portfolio holdings  
        for action in unprocessed_actions:

            # Update portfolio holdings and action as processed
            self.db_access.update_portfolio_holdings_and_action(session, action)
            
            start_time = time.time()
            self.db_access.update_holding_time_series_ffill(session, action, datetime.now().date())
            end_time = time.time()
            runtime = end_time - start_time
            print(f"Runtime for update_holding_time_series_ffill: {runtime:.3f} seconds")

            # Update time series in-kind holding
            start_time = time.time()
            self.db_access.update_holding_time_series_vectorized(session, action, datetime.now().date())
            end_time = time.time()

            runtime = end_time - start_time
            print(f"Runtime for update_holding_time_series: {runtime:.3f} seconds")

        return


# Usage example:
if __name__ == "__main__":
    db_access = DatabaseAccess()
    action_service = ActionService(db_access)
    
    # actions = action_service.read_actions_from_excel("_junk/test_action_data.xlsx")
    # action_service.insert_actions(actions)
    action_service.process_actions()
    

    print('done')
