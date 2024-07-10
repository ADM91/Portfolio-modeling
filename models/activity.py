
from datetime import datetime
from pydantic import BaseModel, validator
from typing import Literal, Optional

import pandas as pd


class Activity(BaseModel):
    Date: datetime
    Asset: str
    Currency: str
    Price: float
    Quantity: float
    Action: Literal['buy', 'sell', 'price change', 'dividend']
    Fee: float
    Account: str
    Platform: Optional[str] = None
    Comment: Optional[str] = None

    @validator('Date', pre=True, always=True)
    def string_to_datetime_coerce(cls, v):
        v = pd.to_datetime(v, dayfirst=True, format='mixed', errors='coerce').date()
        return v
    
    @validator('Quantity')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
    
    @validator('Platform', 'Comment', pre=True, always=True)
    def convert_nan_to_none(cls, v):
        if pd.isna(v):
            return None
        return v

    # @validator('Comment', pre=True, always=True)
    # def convert_nan_to_none(cls, v):
    #     if pd.isna(v):
    #         return None
    #     return v

# from datetime import datetime

# from model.asset import Asset


# class Activity:
#     def __init__(self, asset: Asset, date: datetime, activity_type: str):
#         self.asset = asset
#         self.date = date
#         self.activity_type = activity_type

# class Transaction(Activity):
#     def __init__(self, asset: Asset, date: datetime, quantity: float, price: float, transaction_type: str):
#         super().__init__(asset, date, "Transaction")
#         self.quantity = quantity
#         self.price = price
#         self.transaction_type = transaction_type

# class PriceChange(Activity):
#     def __init__(self, asset: Asset, date: datetime, old_price: float, new_price: float):
#         super().__init__(asset, date, "PriceChange")
#         self.old_price = old_price
#         self.new_price = new_price

# class Dividend(Activity):
#     def __init__(self, asset: Asset, date: datetime, dividend_amount: float):
#         super().__init__(asset, date, "Dividend")
#         self.dividend_amount = dividend_amount