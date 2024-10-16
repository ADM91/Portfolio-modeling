
from datetime import date, timedelta


def calculate_start_date(timeframe: str, end_date: date) -> date:
    if timeframe == "30d":
        return end_date - timedelta(days=30)
    elif timeframe == "90d":
        return end_date - timedelta(days=90)
    elif timeframe == "ytd":
        return date(end_date.year, 1, 1)
    elif timeframe == "1y":
        return end_date - timedelta(days=365)
    elif timeframe == "max":
        return date(1970, 1, 1)  # Arbitrary past date, adjust as needed
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}")