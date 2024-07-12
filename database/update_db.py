
from datetime import datetime, timedelta

from database.access import DatabaseAccess
from database.entities import ActionType, Currency, Asset
from services.yfinance_service import YFinanceService
from config import action_types, currencies, assets



def initialize_database():
    db_access = DatabaseAccess()
    db_access.init_db()
    db_access.insert_if_not_exists(ActionType, action_types)
    db_access.insert_if_not_exists(Currency, currencies)
    db_access.insert_if_not_exists(Asset, assets)


def update_database():
    db_access = DatabaseAccess()
    yf_service = YFinanceService(db_access)

    asset_list = db_access.get_all_assets()

    for asset in asset_list:
        # Get the most recent date in the database for this asset
        last_date = db_access.get_last_price_date(asset.ticker)
        
        # If we have no data, start from 5 years ago, otherwise start from the day after the last date
        start_date = last_date + timedelta(days=1) if last_date else datetime.now() - timedelta(days=5*365)
        end_date = datetime.now()

        # Only fetch data if there's a gap to fill
        if start_date < end_date:
            print(f"Updating {asset.ticker} from {start_date} to {end_date}")
            
            try:
                asset_name, price_data = yf_service.fetch_asset_data(asset.ticker, start_date, end_date)
                
                if not price_data.empty:
                    # Prepare and insert price history data
                    price_history = yf_service.prepare_price_history_for_db(price_data)
                    db_access.add_price_history(asset.ticker, price_history)
                    print(f"Updated {asset.ticker} with {len(price_history)} new entries")
                else:
                    print(f"No new data for {asset.ticker}")
            except Exception as e:
                print(f"Error updating {asset.ticker}: {str(e)}")
        else:
            print(f"{asset.ticker} is up to date")

if __name__ == "__main__":
    # initialize_database()
    update_database()