"""
Improved MetricService using @with_session decorator and Redis caching
Focuses on metric calculations, with tax lot logic moved to TaxLotService
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Optional
from decimal import Decimal
import json
import redis

from database.access import DatabaseAccess, with_session, Session


class MetricService:
    """
    Enhanced MetricService with Redis caching and improved performance
    Uses @with_session decorator and maintains currency-agnostic design
    """

    def __init__(self, db_access: DatabaseAccess, redis_client: Optional[redis.Redis] = None):
        self.db_access = db_access
        self.redis_client = redis_client
        # Cache TTL in seconds (1 hour for price data, 10 minutes for holdings)
        self.price_cache_ttl = 3600
        self.holdings_cache_ttl = 600

    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate consistent cache keys"""
        return f"{prefix}:" + ":".join(str(arg) for arg in args)

    def _cache_dataframe(self, key: str, df: pd.DataFrame, ttl: int) -> None:
        """Cache pandas DataFrame in Redis"""
        if self.redis_client and not df.empty:
            try:
                # Convert to JSON for Redis storage
                self.redis_client.setex(key, ttl, df.to_json(date_format='iso'))
            except Exception as e:
                logging.warning(f"Failed to cache data for key {key}: {e}")

    def _get_cached_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve pandas DataFrame from Redis cache"""
        if not self.redis_client:
            return None
            
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                df = pd.read_json(cached_data, date_format='iso')
                df['date'] = pd.to_datetime(df['date'])
                return df
        except Exception as e:
            logging.warning(f"Failed to retrieve cached data for key {key}: {e}")
        
        return None

    @with_session
    def _get_price_data_with_cache(self, session: Session, asset_id: int, 
                                  start_date: date, end_date: date) -> pd.DataFrame:
        """Get price data with Redis caching"""
        cache_key = self._get_cache_key("prices", asset_id, start_date, end_date)
        
        # Try cache first
        cached_df = self._get_cached_dataframe(cache_key)
        if cached_df is not None:
            return cached_df
        
        # Cache miss - get from database
        df = self.db_access.get_asset_price_history_df(session, asset_id, start_date, end_date)
        
        # Cache the result
        self._cache_dataframe(cache_key, df, self.price_cache_ttl)
        
        return df

    @with_session
    def _get_holdings_data_with_cache(self, session: Session, portfolio_id: int, 
                                     asset_id: int) -> pd.DataFrame:
        """Get holdings data with Redis caching"""
        cache_key = self._get_cache_key("holdings", portfolio_id, asset_id)
        
        # Try cache first
        cached_df = self._get_cached_dataframe(cache_key)
        if cached_df is not None:
            return cached_df
        
        # Cache miss - get from database
        df = self.db_access.get_portfolio_asset_time_series_df(session, portfolio_id, asset_id)
        
        # Cache the result
        self._cache_dataframe(cache_key, df, self.holdings_cache_ttl)
        
        return df

    def preload_cache_for_ui(self, portfolio_ids: List[int], asset_ids: List[int], 
                           currency_ids: List[int], start_date: date, end_date: date) -> None:
        """
        Preload commonly used data into Redis cache for fast UI interactions
        Call this once to enable instant currency switching
        """
        logging.info(f"Preloading cache for {len(portfolio_ids)} portfolios, "
                    f"{len(asset_ids)} assets, {len(currency_ids)} currencies")
        
        # Preload price data for all assets and currencies
        all_asset_ids = list(set(asset_ids + currency_ids))
        for asset_id in all_asset_ids:
            self._get_price_data_with_cache(asset_id, start_date, end_date)
        
        # Preload holdings data
        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                self._get_holdings_data_with_cache(portfolio_id, asset_id)
        
        logging.info("Cache preloading completed")

    def clear_cache(self, pattern: str = "*") -> None:
        """Clear Redis cache entries matching pattern"""
        if self.redis_client:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logging.info(f"Cleared {len(keys)} cache entries")
            except Exception as e:
                logging.warning(f"Failed to clear cache: {e}")

    @with_session
    def get_holdings_in_currency(self, session: Session, portfolio_id: int, asset_id: int, 
                                currency_id: int, start_date: date = date(1970, 1, 1), 
                                end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate holdings in specified currency using cached data for performance
        This is your existing algorithm with caching added
        """
        # Get cached holdings data
        holdings_df = self._get_holdings_data_with_cache(portfolio_id, asset_id)
        
        if holdings_df.empty:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'holding_value'])

        # Filter by date range
        holdings_df = holdings_df[
            (holdings_df['date'] >= pd.to_datetime(start_date)) & 
            (holdings_df['date'] <= pd.to_datetime(end_date))
        ]

        if holdings_df.empty:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'holding_value'])

        # Get cached price data
        asset_prices = self._get_price_data_with_cache(asset_id, start_date, end_date)
        currency_prices = self._get_price_data_with_cache(currency_id, start_date, end_date)

        if asset_prices.empty or currency_prices.empty:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'holding_value'])

        # Your existing vectorized calculation - unchanged
        df = pd.merge(holdings_df, asset_prices[['date', 'close']], on='date', how='left')
        df = pd.merge(df, currency_prices[['date', 'close']], on='date', how='left', 
                     suffixes=('_asset', '_currency'))

        # Handle missing price data
        df = df.dropna(subset=['close_asset', 'close_currency'])
        
        if df.empty:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'holding_value'])

        # Vectorized currency conversion
        df['holding_value'] = df['quantity'] * df['close_asset'] / df['close_currency']
        df['portfolio_id'] = portfolio_id
        df['asset_id'] = asset_id

        return df[['date', 'portfolio_id', 'asset_id', 'holding_value']]

    def get_holdings_in_currency_general(self, portfolio_ids: List[int], asset_ids: List[int], 
                                       currency_id: int, start_date: date = date(1970, 1, 1), 
                                       end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate holdings in currency for multiple portfolios and assets
        Uses caching for better performance with multiple combinations
        """
        all_data = []

        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                holdings_df = self.get_holdings_in_currency(
                    portfolio_id, asset_id, currency_id, start_date, end_date
                )
                if not holdings_df.empty:
                    all_data.append(holdings_df)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            return df.sort_values(['date', 'portfolio_id', 'asset_id']).reset_index(drop=True)
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'holding_value'])

    @with_session
    def get_cash_flow(self, session: Session, portfolio_id: int, asset_id: int, 
                     currency_id: int, start_date: date = date(1970, 1, 1), 
                     end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate cash flow with improved currency conversion using cached price data
        """
        # Check cache for actions data
        actions_cache_key = self._get_cache_key("actions", portfolio_id, asset_id)
        actions_df = self._get_cached_dataframe(actions_cache_key)
        
        if actions_df is None:
            actions = self.db_access.get_buy_sell_actions_by_portfolio_id_asset_id(
                session, portfolio_id, asset_id
            )
            actions_df = pd.read_sql(actions.statement, session.bind)
            # Cache actions data
            self._cache_dataframe(actions_cache_key, actions_df, self.holdings_cache_ttl)
        
        if actions_df.empty:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'cash_flow_daily', 'cash_flow_cumulative'])

        actions_df = actions_df.sort_values(by='date').reset_index(drop=True)
        actions_df['cash_flow'] = 0.0

        # Get unique currencies for batch price loading
        unique_currencies = actions_df['currency_id'].unique()
        
        # Pre-load all needed price data
        action_start = actions_df['date'].min().date()
        action_end = actions_df['date'].max().date()
        
        currency_price_data = {}
        for curr_id in unique_currencies:
            if curr_id != currency_id:
                currency_price_data[curr_id] = self._get_price_data_with_cache(
                    curr_id, action_start, action_end
                )
        
        target_currency_prices = self._get_price_data_with_cache(
            currency_id, action_start, action_end
        )

        # Vectorized currency conversion for each currency group
        for curr_id in unique_currencies:
            mask = actions_df['currency_id'] == curr_id
            if not mask.any():
                continue
                
            curr_actions = actions_df.loc[mask].copy()
            sign = curr_actions['action_type_name'].apply(lambda x: 1 if x == 'buy' else -1)
            
            if curr_id == currency_id:
                # Same currency - no conversion needed
                curr_actions['cash_flow'] = sign * curr_actions['price'] * curr_actions['quantity']
            else:
                # Need currency conversion
                if curr_id in currency_price_data and not target_currency_prices.empty:
                    from_prices = currency_price_data[curr_id]
                    
                    # Merge with price data for conversion
                    curr_actions = pd.merge(curr_actions, from_prices[['date', 'close']], 
                                          left_on='date', right_on='date', how='left')
                    curr_actions = pd.merge(curr_actions, target_currency_prices[['date', 'close']], 
                                          left_on='date', right_on='date', how='left', 
                                          suffixes=('_from', '_to'))
                    
                    # Calculate conversion rate and cash flow
                    curr_actions['conversion_rate'] = curr_actions['close_from'] / curr_actions['close_to']
                    curr_actions['cash_flow'] = (
                        sign * curr_actions['conversion_rate'] * 
                        curr_actions['price'] * curr_actions['quantity']
                    )
                    
                    # Handle NaN values (missing price data)
                    curr_actions['cash_flow'] = curr_actions['cash_flow'].fillna(0)
                else:
                    # Fallback to individual conversion (less efficient)
                    for idx, row in curr_actions.iterrows():
                        try:
                            conversion_rate = self.db_access.get_currency_conversion_on_date(
                                session, row['currency_id'], currency_id, row['date']
                            )
                            curr_actions.at[idx, 'cash_flow'] = (
                                sign.loc[idx] * conversion_rate * row['price'] * row['quantity']
                            )
                        except Exception as e:
                            logging.warning(f"Currency conversion failed for action {row.get('id', 'unknown')}: {e}")
                            curr_actions.at[idx, 'cash_flow'] = 0
            
            # Update main dataframe
            actions_df.loc[mask, 'cash_flow'] = curr_actions['cash_flow'].values

        # Create daily time series
        actions_df = actions_df.groupby(['date', 'portfolio_id', 'asset_id'])['cash_flow'].sum().reset_index()
        actions_df.rename(columns={'cash_flow': 'cash_flow_daily'}, inplace=True)

        # Create full date range
        date_range = pd.date_range(
            start=max(start_date, actions_df['date'].min().date()), 
            end=min(end_date, datetime.now().date()), 
            freq='D'
        )
        
        df = pd.DataFrame(index=date_range)
        df.index.name = 'date'
        
        df = df.merge(actions_df[['date', 'portfolio_id', 'asset_id', 'cash_flow_daily']], 
                     left_index=True, right_on='date', how='left')
        
        df['cash_flow_daily'] = df['cash_flow_daily'].fillna(0)
        df['cash_flow_cumulative'] = df['cash_flow_daily'].cumsum()
        df['portfolio_id'] = portfolio_id
        df['asset_id'] = asset_id
        
        df = df.reset_index()
        
        return df[['date', 'portfolio_id', 'asset_id', 'cash_flow_daily', 'cash_flow_cumulative']]

    def get_cash_flow_general(self, portfolio_ids: List[int], asset_ids: List[int], 
                             currency_id: int, start_date: date = date(1970, 1, 1), 
                             end_date: date = datetime.now().date()) -> pd.DataFrame:
        """Calculate cash flow for multiple portfolios and assets"""
        all_data = []

        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                cash_flow_df = self.get_cash_flow(portfolio_id, asset_id, currency_id, start_date, end_date)
                if not cash_flow_df.empty:
                    all_data.append(cash_flow_df)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            return df.sort_values(['date', 'portfolio_id', 'asset_id']).reset_index(drop=True)
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'cash_flow_daily', 'cash_flow_cumulative'])

    @with_session
    def get_cost_basis_simple(self, session: Session, portfolio_id: int, asset_id: int, currency_id: int,
                             start_date: date = date(1970, 1, 1), 
                             end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Simple cost basis calculation for backwards compatibility
        For accurate tax reporting, use TaxLotService.get_tax_lot_cost_basis_time_series()
        """
        logging.warning("Using simplified cost basis calculation. "
                       "For accurate tax reporting, use TaxLotService.get_tax_lot_cost_basis_time_series()")
        
        # This is your existing implementation - kept for compatibility
        actions = self.db_access.get_buy_sell_actions_by_portfolio_id_asset_id(
            session, portfolio_id, asset_id
        )
        
        actions_df = pd.read_sql(actions.statement, session.bind)
        
        if actions_df.empty:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'cost_basis', 'cost_basis_value', 'asset_quantity'])
        
        actions_df = actions_df.sort_values(by='date').reset_index(drop=True)

        actions_df['quantity'] = actions_df.apply(
            lambda row: row['quantity'] if row['action_type_name'] == 'buy' else -row['quantity'], axis=1
        )
        actions_df['cost'] = 0.0
        actions_df['asset_quantity'] = 0.0
        actions_df['cost_basis_value'] = 0.0

        for i, row in actions_df.iterrows():
            if row['action_type_name'] == 'buy':
                if currency_id != row['currency_id']:
                    try:
                        conversion_rate = self.db_access.get_currency_conversion_on_date(
                            session, row['currency_id'], currency_id, row['date']
                        )
                        actions_df.at[i, 'cost'] = conversion_rate * row['price'] * row['quantity']
                    except Exception as e:
                        logging.warning(f"Currency conversion failed: {e}")
                        actions_df.at[i, 'cost'] = row['price'] * row['quantity']  # Fallback
                else:
                    actions_df.at[i, 'cost'] = row['price'] * row['quantity']
            else:  # sell action
                if i > 0 and actions_df.at[i-1, 'asset_quantity'] > 0:
                    actions_df.at[i, 'cost'] = row['quantity'] * actions_df.at[i-1, 'cost_basis']

            actions_df.at[i, 'asset_quantity'] = actions_df['quantity'][:i+1].sum()
            actions_df.at[i, 'cost_basis_value'] = actions_df['cost'][:i+1].sum()
            
            if actions_df.at[i, 'asset_quantity'] > 0:
                actions_df.at[i, 'cost_basis'] = actions_df.at[i, 'cost_basis_value'] / actions_df.at[i, 'asset_quantity']
            else:
                actions_df.at[i, 'cost_basis'] = 0

        # Aggregate daily cost basis
        actions_agg_df = actions_df.groupby(['date', 'portfolio_id', 'asset_id'])[
            ['cost_basis','cost_basis_value','asset_quantity']
        ].mean().reset_index()

        # Create daily time series
        if not actions_agg_df.empty:
            full_date_range = pd.date_range(
                start=actions_agg_df['date'].min().date(), 
                end=datetime.now().date(), 
                freq='D'
            )
            
            df = pd.DataFrame(index=full_date_range)
            df.index.name = 'date'
            
            df = df.merge(actions_agg_df[['date', 'portfolio_id', 'asset_id', 'cost_basis', 'cost_basis_value', 'asset_quantity']], 
                         left_index=True, right_on='date', how='left')
            
            df['cost_basis'] = df['cost_basis'].ffill().fillna(0)
            df['cost_basis_value'] = df['cost_basis_value'].ffill().fillna(0)
            df['asset_quantity'] = df['asset_quantity'].ffill().fillna(0)
            df['portfolio_id'] = portfolio_id
            df['asset_id'] = asset_id
            
            df = df.reset_index()
            
            # Filter to requested date range
            df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
            
            return df[['date', 'portfolio_id', 'asset_id', 'cost_basis', 'cost_basis_value', 'asset_quantity']]
        
        return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'cost_basis', 'cost_basis_value', 'asset_quantity'])

    def get_unrealized_gain_loss(self, portfolio_id: int, asset_id: int, currency_id: int,
                                start_date: date = date(1970, 1, 1), 
                                end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate unrealized gain/loss using simple cost basis
        For accurate tax calculations, use TaxLotService methods
        """
        # Get cost basis (using simple method for compatibility)
        cost_basis_df = self.get_cost_basis_simple(portfolio_id, asset_id, currency_id, start_date, end_date)
        
        # Get current holdings value
        holdings_df = self.get_holdings_in_currency(portfolio_id, asset_id, currency_id, start_date, end_date)

        if cost_basis_df.empty or holdings_df.empty:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'holding_value', 
                                       'cost_basis_value', 'unrealized_gain_loss', 'unrealized_gain_loss_percentage'])

        # Merge and calculate unrealized gains
        df = pd.merge(cost_basis_df, holdings_df, on=['date', 'portfolio_id', 'asset_id'], how='outer')
        df = df.sort_values('date').ffill()

        df['unrealized_gain_loss'] = df['holding_value'] - df['cost_basis_value']
        
        # Avoid division by zero
        df['unrealized_gain_loss_percentage'] = np.where(
            df['cost_basis_value'] != 0,
            (df['unrealized_gain_loss'] / df['cost_basis_value']) * 100,
            0
        )

        return df[['date', 'portfolio_id', 'asset_id', 'holding_value', 'cost_basis_value', 
                  'unrealized_gain_loss', 'unrealized_gain_loss_percentage']].dropna()

    def get_unrealized_gain_loss_general(self, portfolio_ids: List[int], asset_ids: List[int], 
                                       currency_id: int, start_date: date = date(1970, 1, 1), 
                                       end_date: date = datetime.now().date()) -> pd.DataFrame:
        """Calculate unrealized gain/loss for multiple portfolios and assets"""
        all_data = []

        for portfolio_id in portfolio_ids:
            for asset_id in asset_ids:
                unrealized_df = self.get_unrealized_gain_loss(portfolio_id, asset_id, currency_id, start_date, end_date)
                if not unrealized_df.empty:
                    all_data.append(unrealized_df)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            return df.sort_values(['date', 'portfolio_id', 'asset_id']).reset_index(drop=True)
        else:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'holding_value', 
                                       'cost_basis_value', 'unrealized_gain_loss', 'unrealized_gain_loss_percentage'])

    def get_time_weighted_return_general(self, portfolio_ids: List[int], asset_ids: List[int], 
                                       currency_id: int, start_date: date = date(1970, 1, 1), 
                                       end_date: date = datetime.now().date()) -> pd.DataFrame:
        """Calculate time-weighted returns using cached data for better performance"""
        holdings_value = self.get_holdings_in_currency_general(
            portfolio_ids, asset_ids, currency_id, start_date, end_date
        )
        cash_flow = self.get_cash_flow_general(
            portfolio_ids, asset_ids, currency_id, start_date, end_date
        )

        if holdings_value.empty or cash_flow.empty:
            return pd.DataFrame(columns=['date', 'daily_adjusted_return', 'time_weighted_return'])

        # Merge and aggregate
        data = pd.merge(cash_flow, holdings_value, on=['date', 'portfolio_id', 'asset_id'], how='outer')
        data_agg = self.aggregate_on_date(data, ['cash_flow_daily', 'holding_value'])

        if len(data_agg) < 2:
            return pd.DataFrame(columns=['date', 'daily_adjusted_return', 'time_weighted_return'])

        # Calculate time-weighted returns with improved error handling
        holding_values = data_agg['holding_value'].values
        cash_flows = data_agg['cash_flow_daily'].values
        
        # Avoid division by zero and handle edge cases
        prev_values = holding_values[:-1]
        prev_values = np.where(prev_values == 0, np.nan, prev_values)
        
        adjusted_daily_returns = (holding_values[1:] - cash_flows[1:]) / prev_values
        adjusted_daily_returns = np.nan_to_num(adjusted_daily_returns, nan=0.0, posinf=0.0, neginf=0.0)
        adjusted_daily_returns = np.insert(adjusted_daily_returns, 0, 0)  # First day has no return
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + adjusted_daily_returns)
        
        data_agg['daily_adjusted_return'] = adjusted_daily_returns
        data_agg['time_weighted_return'] = cumulative_returns

        return data_agg[['date', 'daily_adjusted_return', 'time_weighted_return']]

    def get_sharpe_ratio(self, portfolio_ids: List[int], asset_ids: List[int], currency_id: int,
                        risk_free_rate: float = 0.0, rolling_window_periods: int = 365, 
                        periods_per_year: int = 365, start_date: date = date(1970, 1, 1), 
                        end_date: date = datetime.now().date()) -> pd.DataFrame:
        """Calculate Sharpe ratio with improved error handling"""
        twr_df = self.get_time_weighted_return_general(
            portfolio_ids, asset_ids, currency_id, start_date, end_date
        )

        if twr_df.empty or len(twr_df) < rolling_window_periods:
            return pd.DataFrame(columns=['date', 'rolling_sharpe_ratio'])

        returns = twr_df['daily_adjusted_return']
        
        # Calculate rolling statistics
        rolling_mean = returns.rolling(window=rolling_window_periods, min_periods=rolling_window_periods//2).mean()
        rolling_std = returns.rolling(window=rolling_window_periods, min_periods=rolling_window_periods//2).std()
        
        # Annualize
        annualized_return = rolling_mean * periods_per_year
        annualized_std = rolling_std * np.sqrt(periods_per_year)
        
        # Calculate Sharpe ratio with improved error handling
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = np.where(
            (annualized_std != 0) & (~np.isnan(annualized_std)) & (~np.isnan(excess_return)), 
            excess_return / annualized_std, 
            0
        )

        df = pd.DataFrame({
            'date': twr_df['date'],
            'rolling_sharpe_ratio': sharpe_ratio
        })

        return df[['date', 'rolling_sharpe_ratio']].dropna()

    def aggregate_on_date(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Aggregate dataframe by date with error handling"""
        if df.empty:
            return pd.DataFrame(columns=['date'] + columns)
        
        try:
            aggregated_df = df.groupby('date')[columns].sum().reset_index()
            return aggregated_df
        except Exception as e:
            logging.error(f"Error aggregating data: {e}")
            return pd.DataFrame(columns=['date'] + columns)

    # Utility methods for cache management
    def invalidate_cache_for_portfolio_asset(self, portfolio_id: int, asset_id: int) -> None:
        """Invalidate cache entries for a specific portfolio/asset combination"""
        patterns = [
            f"holdings:{portfolio_id}:{asset_id}",
            f"actions:{portfolio_id}:{asset_id}"
        ]
        
        for pattern in patterns:
            self.clear_cache(pattern)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get Redis cache statistics"""
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        try:
            info = self.redis_client.info('memory')
            keyspace = self.redis_client.info('keyspace')
            
            total_keys = 0
            for db_key, db_info in keyspace.items():
                if 'keys' in db_info:
                    total_keys += db_info['keys']
            
            return {
                "total_keys": total_keys,
                "memory_used_mb": round(info['used_memory'] / 1024 / 1024, 2),
                "memory_peak_mb": round(info['used_memory_peak'] / 1024 / 1024, 2)
            }
        except Exception as e:
            return {"error": str(e)}
