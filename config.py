
# # Activity file path
# PATH_ACTIVITY = "portfolio_activity.xlsx"

# # Configuration for assets and metrics
# ACCOUNTS = ['Alexander', 'AM ehf.']
# ASSETS = ["BTC", "ETH", "MATIC", "ADA", "DOT", "LINK", "SOL", "ISK", "EUR"]
# DENOMINATIONS = ["In-kind", "USD", "ISK", "BTC"]
# METRICS = ["Balance", "Cost Basis", "ROI"]

# # Map tickers to assets
# TICKER_ASSET_MAP = {"BTC-USD":"BTC", "ETH-USD":"ETH", "MATIC-USD":"MATIC", "ADA-USD":"ADA", "DOT-USD":"DOT", "LINK-USD":"LINK", "SOL-USD":"SOL", "ISK=X":"ISK", "EUR=X":"EUR"}


# Data to insert
action_types = [
    {"name": "buy"},
    {"name": "sell"},
    {"name": "dividend"}
]

assets = [
    {"ticker": "ISK=X", "code": "ISK", "name": "Icelandic krona", "is_currency": True, "is_inverted": True},
    {"ticker": "EUR=X", "code": "EUR", "name": "Euro", "is_currency": True, "is_inverted": True},
    {"ticker": "BTC-USD", "code": "BTC", "name": "Bitcoin", "is_currency": True, "is_inverted": False},
    {"ticker": "ETH-USD", "code": "ETH", "name": "Ethereum", "is_currency": False, "is_inverted": False},
    {"ticker": "MATIC-USD", "code": "MATIC", "name": "Polygon", "is_currency": False, "is_inverted": False}
]


portfolios = [
    {"name": "Alexander", "owner": "Alexander"},
    {"name": "AM ehf.", "owner": "AM ehf."},
    {"name": "Pauline", "owner": "Pauline"},
    {"name": "Óliver", "owner": "Óliver"},
]