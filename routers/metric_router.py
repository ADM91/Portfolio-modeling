
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime

from startup import db_update
from services.metric_service import MetricService
from database.access import DatabaseAccess
from utils.date_utils import calculate_start_date


router = APIRouter(
    prefix="/metrics",
    tags=["metrics"]
)

metric_service = MetricService(DatabaseAccess())

@router.get("/time-weighted-return")
async def get_time_weighted_return(
    currency_id: int,
    portfolio_ids: List[int] = Query(),
    asset_ids: List[int] = Query(),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: Optional[str] = Query(default="30d", enum=["30d", "90d", "ytd", "1y", "max"]),
):
    try:
        db_update()  # TODO: do this more efficiently, check if db is up to date before running
        end = datetime.now().date()
        if timeframe:
            start = calculate_start_date(timeframe, end)
        elif start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            raise HTTPException(status_code=400, detail="Either timeframe or start_date and end_date must be provided")

        result = metric_service.get_time_weighted_return_general(portfolio_ids, asset_ids, currency_id, start, end).round(5)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/unrealized-gain-loss")
async def get_unrealized_gain_loss(
    currency_id: int,
    portfolio_ids: List[int] = Query(),
    asset_ids: List[int] = Query(),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: Optional[str] = Query(default="30d", enum=["30d", "90d", "ytd", "1y", "max"]),
):
    try:
        db_update()  # TODO: do this more efficiently, check if db is up to date before running
        end = datetime.now().date()
        if timeframe:
            start = calculate_start_date(timeframe, end)
        elif start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            raise HTTPException(status_code=400, detail="Either timeframe or start_date and end_date must be provided")

        result = metric_service.get_unrealized_gain_loss_general(portfolio_ids, asset_ids, currency_id, start, end).round(1)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sharpe-ratio")
async def get_rolling_sharpe_ratio(
    currency_id: int,
    portfolio_ids: List[int] = Query(),
    asset_ids: List[int] = Query(),
    risk_free_rate: float=0.0,
    rolling_window_periods: int=365,
    periods_per_year: int=365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: Optional[str] = Query(default="30d", enum=["30d", "90d", "ytd", "1y", "max"]),
):
    try:
        db_update()  # TODO: do this more efficiently, check if db is up to date before running
        end = datetime.now().date()
        if timeframe:
            start = calculate_start_date(timeframe, end)
        elif start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            raise HTTPException(status_code=400, detail="Either timeframe or start_date and end_date must be provided")

        result = metric_service.get_sharpe_ratio(portfolio_ids, asset_ids, currency_id, risk_free_rate, rolling_window_periods, periods_per_year, start, end).round(3)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/value-invested")
async def get_value_invested(
    currency_id: int,
    portfolio_ids: List[int] = Query(),
    asset_ids: List[int] = Query(), 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: Optional[str] = Query(default="30d", enum=["30d", "90d", "ytd", "1y", "max"]),
):
    try:
        db_update()  # TODO: do this more efficiently, check if db is up to date before running
        end = datetime.now().date()
        if timeframe:
            start = calculate_start_date(timeframe, end)
        elif start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            raise HTTPException(status_code=400, detail="Either timeframe or start_date and end_date must be provided")

        result = metric_service.get_cash_flow_general(portfolio_ids, asset_ids, currency_id, start, end).round(1)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))