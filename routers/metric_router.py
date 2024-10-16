
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime

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
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: Optional[str] = Query(default="30d", enum=["30d", "90d", "ytd", "1y", "max"]),
    portfolio_ids: List[int] = Query(),
    asset_ids: List[int] = Query(),
):
    try:
        end = datetime.now().date()
        
        if timeframe:
            start = calculate_start_date(timeframe, end)
        elif start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            raise HTTPException(status_code=400, detail="Either timeframe or start_date and end_date must be provided")

        result = metric_service.get_time_weighted_return_general(portfolio_ids, asset_ids, currency_id, start, end)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sharpe-ratio")
async def get_sharpe_ratio(
    portfolio_ids: List[int],
    asset_ids: List[int],
    currency_id: int,
    start_date: str,
    end_date: str,
    risk_free_rate: float = 0.02
):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        result = metric_service.get_sharpe_ratio(portfolio_ids, asset_ids, currency_id, start, end, risk_free_rate)
        return {"sharpe_ratio": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/unrealized-gain-loss")
async def get_unrealized_gain_loss(
    portfolio_ids: List[int],
    asset_ids: List[int],
    currency_id: int
):
    try:
        result = metric_service.get_unrealized_gain_loss_general(portfolio_ids, asset_ids, currency_id)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
