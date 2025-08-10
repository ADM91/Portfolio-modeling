"""
API Client for Portfolio Dashboard
Handles communication with FastAPI backend
"""
import requests
import pandas as pd
import streamlit as st
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import logging


class PortfolioAPIClient:
    """Client for interacting with the Portfolio API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> List[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API server. Please ensure the FastAPI server is running on http://localhost:8000")
            return []
        except requests.exceptions.Timeout:
            st.error("⏱️ API request timed out. Please try again.")
            return []
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API Error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return []
    
    def _prepare_params(self, currency_id: int, portfolio_ids: List[int], 
                       asset_ids: List[int], timeframe: str = "30d",
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> Dict[str, Any]:
        """Prepare API parameters"""
        params = {
            "currency_id": currency_id,
            "timeframe": timeframe,
            "portfolio_ids": portfolio_ids,
            "asset_ids": asset_ids
        }
            
        if start_date and end_date:
            params["start_transaction_datetime"] = start_date
            params["end_transaction_datetime"] = end_date
            params.pop("timeframe", None)
            
        return params
    
    @st.cache_data(ttl=300)
    def get_holdings_value(_self, currency_id: int, portfolio_ids: List[int], 
                          asset_ids: List[int], timeframe: str = "30d",
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Get holdings value time series"""
        params = _self._prepare_params(currency_id, portfolio_ids, asset_ids, 
                                      timeframe, start_date, end_date)
        
        data = _self._make_request("/api/v1/metrics/holdings-value", params)
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        if 'transaction_datetime' in df.columns:
            df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
        return df
    
    @st.cache_data(ttl=300)
    def get_value_invested(_self, currency_id: int, portfolio_ids: List[int], 
                          asset_ids: List[int], timeframe: str = "30d",
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Get value invested time series"""
        params = _self._prepare_params(currency_id, portfolio_ids, asset_ids, 
                                      timeframe, start_date, end_date)
        
        data = _self._make_request("/api/v1/metrics/value-invested", params)
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        if 'transaction_datetime' in df.columns:
            df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
        return df
    
    @st.cache_data(ttl=300)
    def get_time_weighted_return(_self, currency_id: int, portfolio_ids: List[int], 
                                asset_ids: List[int], timeframe: str = "30d",
                                start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """Get time-weighted return time series"""
        params = _self._prepare_params(currency_id, portfolio_ids, asset_ids, 
                                      timeframe, start_date, end_date)
        
        data = _self._make_request("/api/v1/metrics/time-weighted-return", params)
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        if 'transaction_datetime' in df.columns:
            df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
        return df
    
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def get_available_assets(_self) -> List[Dict[str, Any]]:
        """Get list of available assets for base currency selection"""
        data = _self._make_request("/api/v1/data/assets", {})
        return data if data else []
    
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def get_available_portfolios(_self) -> List[Dict[str, Any]]:
        """Get list of available portfolios"""
        data = _self._make_request("/api/v1/data/portfolios", {})
        return data if data else []
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_available_portfolio_assets(_self, portfolio_ids: List[int]) -> List[Dict[str, Any]]:
        """Get assets available in selected portfolios"""
        if not portfolio_ids:
            return []
        
        params = {"portfolio_ids": portfolio_ids}
        data = _self._make_request("/api/v1/data/portfolio-assets", params)
        return data if data else []
