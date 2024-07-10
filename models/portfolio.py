
from model.asset import Asset

class Portfolio:
    def __init__(self):
        self.assets = {}  # Key: ticker, Value: (Asset, quantity)
        self.cash_holdings = 0.0

    def add_asset(self, asset: Asset, quantity: float):
        if asset.ticker in self.assets:
            self.assets[asset.ticker][1] += quantity
        else:
            self.assets[asset.ticker] = (asset, quantity)

    def remove_asset(self, ticker: str, quantity: float = None):
        if ticker in self.assets:
            if quantity is None or self.assets[ticker][1] <= quantity:
                del self.assets[ticker]
            else:
                self.assets[ticker][1] -= quantity

    def update_quantity(self, ticker: str, quantity: float):
        if ticker in self.assets:
            self.assets[ticker][1] = quantity

    def get_total_value(self):
        # Placeholder for total value calculation; this would require price data
        return sum(asset[1] for asset in self.assets.values()) + self.cash_holdings
