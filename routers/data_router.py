"""
Data Router - Endpoints for getting portfolios, assets, etc.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from sqlalchemy import distinct

from database.access import DatabaseAccess, session_scope
from database.entities import Portfolio, Asset, PortfolioHoldingsTimeSeries

router = APIRouter(
    prefix="/data",
    tags=["data"]
)

db_access = DatabaseAccess()

@router.get("/assets")
async def get_all_assets():
    """Get all available assets for base currency selection"""
    try:
        with session_scope() as session:
            assets = session.query(Asset).all()
            
            result = []
            for asset in assets:
                # This categorization does not belong here, should be defined in database or domain model
                asset_type = "Currency" if asset.is_currency else "Asset"
                if asset.ticker.upper() in ["BTC", "ETH", "ADA", "DOT"]:  # Common crypto symbols
                    asset_type = "Cryptocurrency"
                elif asset.ticker.upper() in ["GOLD", "SILVER", "OIL"]:  # Common commodities
                    asset_type = "Commodity"
                elif asset.is_currency:
                    asset_type = "Fiat"
                else:
                    asset_type = "Stock/ETF"
                
                result.append({
                    "id": asset.id,
                    "symbol": asset.ticker,
                    "name": asset.name or asset.ticker,
                    "type": asset_type
                })
            
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching assets: {str(e)}")


@router.get("/portfolios")
async def get_all_portfolios():
    """Get all available portfolios"""
    try:
        with session_scope() as session:
            portfolios = session.query(Portfolio).all()
            
            result = []
            for portfolio in portfolios:
                result.append({
                    "id": portfolio.id,
                    "name": portfolio.name,
                    "owner": portfolio.owner or "Unknown"
                })
            
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching portfolios: {str(e)}")


@router.get("/portfolio-assets")
async def get_portfolio_assets(portfolio_ids: List[int] = Query()):
    """Get assets available in selected portfolios"""
    try:
        if not portfolio_ids:
            return []
        
        with session_scope() as session:
            # Get distinct assets from the selected portfolios
            assets_query = session.query(Asset).join(
                PortfolioHoldingsTimeSeries, Asset.id == PortfolioHoldingsTimeSeries.asset_id
            ).filter(
                PortfolioHoldingsTimeSeries.portfolio_id.in_(portfolio_ids)
            ).distinct()
            
            assets = assets_query.all()
            
            result = []
            for asset in assets:
                # This categorization does not belong here, should be defined in database or domain model
                asset_type = "Stock/ETF"
                if asset.is_currency:
                    asset_type = "Currency"
                if asset.ticker.upper() in ["BTC", "ETH", "ADA", "DOT"]:
                    asset_type = "Cryptocurrency"
                elif asset.ticker.upper() in ["GOLD", "SILVER", "OIL"]:
                    asset_type = "Commodity"
                
                result.append({
                    "id": asset.id,
                    "symbol": asset.ticker,
                    "name": asset.name or asset.ticker,
                    "type": asset_type
                })
            
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio assets: {str(e)}")
