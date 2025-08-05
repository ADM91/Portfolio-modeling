"""
Backward-compatible wrapper for the original YFinanceService.
This maintains the original interface while using the new generalized DataAcquisitionService internally.
"""

from services.data_acquisition_service import DataAcquisitionService
from services.data_providers import YFinanceProvider
from database.access import DatabaseAccess


class YFinanceService:
    """
    Backward-compatible wrapper for YFinanceService.
    Delegates to the new DataAcquisitionService with YFinanceProvider.
    """

    def __init__(self, db_access: DatabaseAccess) -> None:
        # Use the new generalized service with Yahoo Finance provider
        self._service = DataAcquisitionService(db_access, YFinanceProvider())

    def fetch_asset_data(self, ticker: str, start_date=None, end_date=None):
        """
        Fetches asset data from yFinance API.
        Delegates to the new DataAcquisitionService.
        """
        return self._service.fetch_asset_data(ticker, start_date, end_date)

    def prepare_asset_for_db(self, ticker: str, asset_name: str):
        """
        Prepares asset data for database insertion.
        Delegates to the new DataAcquisitionService.
        """
        return self._service.prepare_asset_for_db(ticker, asset_name)

    def prepare_price_history_for_db(self, price_data):
        """
        Prepares price history data for database insertion.
        Delegates to the new DataAcquisitionService.
        """
        return self._service.prepare_price_history_for_db(price_data)

    def update_db_with_asset_data(self):
        """
        Updates database with asset data.
        Delegates to the new DataAcquisitionService.
        """
        return self._service.update_db_with_asset_data()


# Usage example - maintains backward compatibility
if __name__ == "__main__":
    from database.access import DatabaseAccess

    yf_service = YFinanceService(DatabaseAccess())
    yf_service.update_db_with_asset_data()

    print('done')
