
import pandas as pd

def get_conversion_rate(price_data_df, date, from_currency, to_currency):
    from_currnecy_in_usd = price_data_df.loc[date][from_currency]
    to_currency_in_usd = price_data_df.loc[date][to_currency]
    conversion_rate =  from_currnecy_in_usd / to_currency_in_usd
    return conversion_rate