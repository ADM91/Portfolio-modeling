
import pandas as pd

from database.access import DatabaseAccess
from config import DENOMINATIONS, METRICS


class Asset:
    def __init__(self, name, ticker_symbol, initial_balance=0):
        self.name = name
        self.ticker_symbol = ticker_symbol

        self.db_access = DatabaseAccess()
        self.db_access.add_asset(ticker_symbol, name)

        # # DataFrame for logging actions
        # self.actions = pd.DataFrame(columns=['Date', 'ActionType', 'Currency', 'Amount', 'Remarks'])
        # # DataFrame for tracking balances in various denominations
        # self.balances = pd.DataFrame(index=pd.to_datetime([]), columns=['InKind', *DENOMINATIONS])
        # # DataFrame for tracking different metrics over time (e.g., ROI, volatility)
        # self.metrics = pd.DataFrame(index=pd.to_datetime([]), columns=METRICS)

    def add_action(self, action):
        """
        Updates balances, metrics, and logs the transaction action.
        """

        # assert that the transaction has a valid currency (one that exists in the balances DataFrame)
        assert action.currency in self.balances.columns

        # Log the transaction
        self._log_action(date=action.date, action_type=action.action_type, currency=action.currency, amount=action.amount, remarks=action.comment)
        # Update balance
        self._update_balance(date=action.date, currency=action.currency, amount=action.amount)
        # Update metrics
        self._update_metrics(date=action.date)

    def _update_balance(self, date, currency, amount):
        """
        Update the balance DataFrame for a given currency and amount.
        """
        if date in self.balances.index:
            self.balances.loc[date, currency] += amount
        else:
            # Add a new row for the date if it doesn't exist
            new_row = pd.Series(name=date, data={currency: amount})
            self.balances = self.balances.append(new_row, ignore_index=False).fillna(0)
        self.balances.ffill(inplace=True)

    def _update_metrics(self, date):
        """
        Recalculate metrics based on the latest data.
        """
        pass  # Implement metric calculation and update logic here

    def _log_action(self, date, action_type, currency, amount, remarks=''):
        """
        Logs actions affecting the asset.
        """
        new_action = {'Date': date, 'ActionType': action_type, 'Currency': currency, 'Amount': amount, 'Remarks': remarks}
        self.actions = self.actions.append(new_action, ignore_index=True)
