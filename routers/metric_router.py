from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import datetime
from sqlalchemy.orm import Session
from dependencies import get_db
from services.metric_service import MetricService

router = APIRouter(
    prefix="/metrics",
    tags=["metrics"]
)

metric_service = MetricService()

@router.get("/time-weighted-return")
async def get_time_weighted_return(
    portfolio_ids: List[int],
    asset_ids: List[int],
    currency_id: int,
    start_date: str,
    end_date: str,
    db: Session = Depends(get_db)
):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        result = metric_service.get_time_weighted_return_general(db, portfolio_ids, asset_ids, currency_id, start, end)
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
    risk_free_rate: float = 0.02,
    db: Session = Depends(get_db)
):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        result = metric_service.get_sharpe_ratio(db, portfolio_ids, asset_ids, currency_id, start, end, risk_free_rate)
        return {"sharpe_ratio": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/unrealized-gain-loss")
async def get_unrealized_gain_loss(
    portfolio_ids: List[int],
    asset_ids: List[int],
    currency_id: int,
    db: Session = Depends(get_db)
):
    try:
        result = metric_service.get_unrealized_gain_loss_general(db, portfolio_ids, asset_ids, currency_id)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
