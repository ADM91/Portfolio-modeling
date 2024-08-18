# action_service.py

import logging
import pandas as pd
from typing import List, Dict, Optional

from database.access_v2 import (
    session_scope, with_session, get_portfolio_by_name, 
    get_action_type_by_name, get_asset_by_code, insert_if_not_exists
)
from database.entities import Action, Portfolio, ActionType, Asset


@with_session
def read_actions_from_excel_to_action_list(session, excel_path: str) -> list[Action]:
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
            portfolio = get_portfolio_by_name(session, row['Portfolio'])
            if portfolio:
                row['portfolio_id'] = portfolio.id
            else:
                raise ValueError(f"Invalid portfolio name: {row['portfolio_name']}")

            # Get action_type_id based on action_type name
            action_type = get_action_type_by_name(session, row['Action'])
            if action_type:
                row['action_type_id'] = action_type.id
            else:
                raise ValueError(f"Invalid action type: {row['action_type']}")

            # Get asset_id based on asset symbol
            asset = get_asset_by_code(session, row['Asset'])
            if asset:
                row['asset_id'] = asset.id
            else:
                raise ValueError(f"Invalid asset symbol: {row['Asset']}")

            # Get currency_id based on currency symbol
            currency = get_asset_by_code(session, row['Currency'])
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

            # convert any nan values to None
            row = row.where((pd.notnull(row)), None)

            # Convert the row to a dictionary and then create an Action instance
            action_list.append(row.to_dict())

        except Exception as e:
            logging.error(f"Error converting row to Action: {e}")
            logging.error(f"action: {row}")
        
    return action_list

@with_session
def insert_actions(session, actions_data: List[dict]):
    """
    Inserts an action into the database if it doesn't already exist.
    """
    filter_fields=['portfolio_id', 'action_type_id', 'date', 'asset_id', 'currency_id', 'price', 'quantity']
    insert_if_not_exists(session, Action, actions_data, filter_fields)

def update_action():
    pass

def delete_action():
    pass


if __name__ == "__main__":
    actions = read_actions_from_excel_to_action_list("_junk/test_action_data.xlsx")
    insert_actions(actions)

    print('done')