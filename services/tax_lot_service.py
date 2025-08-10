"""
Tax Lot Service for managing tax lots and tax calculations
"""
import logging
from datetime import datetime, date
from typing import List, Dict, Optional
from decimal import Decimal
import pandas as pd
from sqlalchemy.orm import Session

from database.access import DatabaseAccess, with_session
from database.entities import TaxLot, TaxLotTransaction, Action, ActionType


class TaxLotService:
    """
    Service for managing tax lots and tax-related calculations
    Handles FIFO, LIFO, and specific identification methods
    """

    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access

    @with_session
    def create_tax_lots_from_buy_action(self, session: Session, action: Action) -> List[TaxLot]:
        """
        Create tax lots when processing buy actions
        
        Args:
            session: Database session
            action: Buy action to create tax lots from
            
        Returns:
            List of created tax lots
        """
        if action.action_type.name != 'buy':
            return []

        # Create unique lot identifier
        lot_identifier = f"{action.asset.ticker}_{action.date.strftime('%Y%m%d_%H%M%S')}_{action.id}"
        
        tax_lot = TaxLot(
            portfolio_id=action.portfolio_id,
            asset_id=action.asset_id,
            lot_identifier=lot_identifier,
            acquisition_date=action.date,
            original_quantity=action.quantity,
            remaining_quantity=action.quantity,
            cost_basis_per_unit=action.price,
            currency_id=action.currency_id,
            is_closed=False
        )
        
        session.add(tax_lot)
        session.flush()

        # Create transaction linking action to tax lot
        tax_lot_transaction = TaxLotTransaction(
            action_id=action.id,
            tax_lot_id=tax_lot.id,
            quantity_change=action.quantity,
            processed_at=datetime.utcnow()
        )
        
        session.add(tax_lot_transaction)
        
        logging.info(f"Created tax lot {lot_identifier} for {action.quantity} {action.asset.ticker}")
        
        return [tax_lot]

    @with_session  
    def process_sell_action_with_tax_lots(self, session: Session, action: Action, 
                                        method: str = 'fifo') -> List[TaxLotTransaction]:
        """
        Process sell actions using specified tax lot method
        
        Args:
            session: Database session
            action: Sell action to process
            method: Tax lot method ('fifo', 'lifo', 'average_cost', 'specific_id')
            
        Returns:
            List of tax lot transactions created
        """
        if action.action_type.name != 'sell':
            return []

        # Get available tax lots
        available_lots_query = session.query(TaxLot).filter(
            TaxLot.portfolio_id == action.portfolio_id,
            TaxLot.asset_id == action.asset_id,
            TaxLot.remaining_quantity > 0,
            TaxLot.is_closed == False
        )

        # Apply ordering based on method
        if method.lower() == 'fifo':
            available_lots = available_lots_query.order_by(TaxLot.acquisition_date.asc()).all()
        elif method.lower() == 'lifo':
            available_lots = available_lots_query.order_by(TaxLot.acquisition_date.desc()).all()
        elif method.lower() == 'average_cost':
            # For average cost, we'll use FIFO ordering but calculate average cost basis
            available_lots = available_lots_query.order_by(TaxLot.acquisition_date.asc()).all()
        else:
            raise ValueError(f"Unsupported tax lot method: {method}")

        if not available_lots:
            raise ValueError(f"No available tax lots for sale of {action.quantity} {action.asset.ticker}")

        # Calculate total available quantity
        total_available = sum(lot.remaining_quantity for lot in available_lots)
        if total_available < action.quantity:
            raise ValueError(f"Insufficient tax lots: need {action.quantity}, have {total_available}")

        transactions = []
        remaining_to_sell = action.quantity
        
        if method.lower() == 'average_cost':
            # Calculate weighted average cost basis
            total_cost_basis = sum(lot.remaining_quantity * lot.cost_basis_per_unit for lot in available_lots)
            total_quantity = sum(lot.remaining_quantity for lot in available_lots)
            avg_cost_basis = total_cost_basis / total_quantity if total_quantity > 0 else Decimal('0')
            
            # Process sale using average cost
            for lot in available_lots:
                if remaining_to_sell <= 0:
                    break
                    
                sell_from_lot = min(remaining_to_sell, lot.remaining_quantity)
                
                # Use average cost basis instead of specific lot cost
                proceeds = sell_from_lot * action.price
                cost_basis = sell_from_lot * avg_cost_basis
                
                # Convert to same currency for gain/loss calculation
                if lot.currency_id != action.currency_id:
                    conversion_rate = self.db_access.get_currency_conversion_on_date(
                        session, lot.currency_id, action.currency_id, action.date
                    )
                    cost_basis = cost_basis * Decimal(str(conversion_rate))
                
                realized_gain_loss = proceeds - cost_basis
                
                transaction = self._create_tax_lot_transaction(
                    action, lot, sell_from_lot, proceeds, realized_gain_loss
                )
                
                session.add(transaction)
                transactions.append(transaction)
                
                # Update lot
                lot.remaining_quantity -= sell_from_lot
                if lot.remaining_quantity == 0:
                    lot.is_closed = True
                    
                remaining_to_sell -= sell_from_lot
        else:
            # FIFO or LIFO processing
            for lot in available_lots:
                if remaining_to_sell <= 0:
                    break
                    
                sell_from_lot = min(remaining_to_sell, lot.remaining_quantity)
                
                proceeds = sell_from_lot * action.price
                cost_basis = sell_from_lot * lot.cost_basis_per_unit
                
                # Convert to same currency for gain/loss calculation
                if lot.currency_id != action.currency_id:
                    conversion_rate = self.db_access.get_currency_conversion_on_date(
                        session, lot.currency_id, action.currency_id, action.date
                    )
                    cost_basis = cost_basis * Decimal(str(conversion_rate))
                
                realized_gain_loss = proceeds - cost_basis
                
                transaction = self._create_tax_lot_transaction(
                    action, lot, sell_from_lot, proceeds, realized_gain_loss
                )
                
                session.add(transaction)
                transactions.append(transaction)
                
                # Update lot
                lot.remaining_quantity -= sell_from_lot
                if lot.remaining_quantity == 0:
                    lot.is_closed = True
                    
                remaining_to_sell -= sell_from_lot

        if remaining_to_sell > 0:
            raise ValueError(f"Failed to process complete sale: {remaining_to_sell} remaining")

        logging.info(f"Processed sale of {action.quantity} {action.asset.ticker} using {method.upper()}")
        
        return transactions

    def _create_tax_lot_transaction(self, action: Action, lot: TaxLot, quantity: Decimal, 
                                   proceeds: Decimal, realized_gain_loss: Decimal) -> TaxLotTransaction:
        """Helper method to create tax lot transaction"""
        return TaxLotTransaction(
            action_id=action.id,
            tax_lot_id=lot.id,
            quantity_change=-quantity,  # Negative for sales
            proceeds=proceeds,
            realized_gain_loss=realized_gain_loss,
            processed_at=datetime.utcnow()
        )

    @with_session
    def get_tax_lot_cost_basis_time_series(self, session: Session, portfolio_id: int, 
                                          asset_id: int, currency_id: int,
                                          start_date: date = date(1970, 1, 1), 
                                          end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Calculate cost basis time series using actual tax lots
        
        Args:
            session: Database session
            portfolio_id: Portfolio ID
            asset_id: Asset ID  
            currency_id: Currency to express cost basis in
            start_date: Start date for time series
            end_date: End date for time series
            
        Returns:
            DataFrame with daily cost basis information
        """
        # Get all tax lots for this portfolio/asset
        tax_lots = session.query(TaxLot).filter(
            TaxLot.portfolio_id == portfolio_id,
            TaxLot.asset_id == asset_id
        ).order_by(TaxLot.acquisition_date).all()

        if not tax_lots:
            return pd.DataFrame(columns=['date', 'portfolio_id', 'asset_id', 'cost_basis', 
                                       'cost_basis_value', 'asset_quantity'])

        # Get all tax lot transactions
        lot_ids = [lot.id for lot in tax_lots]
        transactions = session.query(TaxLotTransaction).filter(
            TaxLotTransaction.tax_lot_id.in_(lot_ids)
        ).order_by(TaxLotTransaction.processed_at).all()

        # Create date range
        earliest_date = min(lot.acquisition_date for lot in tax_lots).date()
        date_range = pd.date_range(
            start=max(start_date, earliest_date),
            end=end_date,
            freq='D'
        )

        # Build daily cost basis time series
        daily_data = []
        
        for current_date in date_range:
            current_datetime = datetime.combine(current_date, datetime.min.time())
            
            total_quantity = Decimal('0')
            total_cost_basis_value = Decimal('0')
            
            for lot in tax_lots:
                if lot.acquisition_date <= current_datetime:
                    # Calculate current state of this lot
                    lot_transactions = [t for t in transactions 
                                      if t.tax_lot_id == lot.id and t.processed_at <= current_datetime]
                    
                    current_quantity = lot.original_quantity
                    for trans in lot_transactions:
                        current_quantity += trans.quantity_change
                    
                    if current_quantity > 0:
                        # Convert cost basis to target currency if needed
                        if lot.currency_id == currency_id:
                            cost_per_unit = lot.cost_basis_per_unit
                        else:
                            conversion_rate = self.db_access.get_currency_conversion_on_date(
                                session, lot.currency_id, currency_id, lot.acquisition_date
                            )
                            cost_per_unit = lot.cost_basis_per_unit * Decimal(str(conversion_rate))
                        
                        total_quantity += current_quantity
                        total_cost_basis_value += current_quantity * cost_per_unit
            
            # Calculate weighted average cost basis
            if total_quantity > 0:
                avg_cost_basis = total_cost_basis_value / total_quantity
            else:
                avg_cost_basis = Decimal('0')
            
            daily_data.append({
                'date': current_date,
                'portfolio_id': portfolio_id,
                'asset_id': asset_id,
                'cost_basis': float(avg_cost_basis),
                'cost_basis_value': float(total_cost_basis_value),
                'asset_quantity': float(total_quantity)
            })

        return pd.DataFrame(daily_data)

    @with_session
    def get_realized_gains_report(self, session: Session, portfolio_id: int, 
                                 asset_id: Optional[int] = None, currency_id: int = 1,
                                 start_date: date = date(1970, 1, 1), 
                                 end_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Generate realized gains/losses report for tax purposes
        
        Args:
            session: Database session
            portfolio_id: Portfolio ID
            asset_id: Optional asset ID (None for all assets)
            currency_id: Currency to express gains in
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            DataFrame with realized gain/loss details
        """
        # Build query for sell transactions with realized gains
        query = session.query(TaxLotTransaction, Action, TaxLot).join(
            Action, TaxLotTransaction.action_id == Action.id
        ).join(
            TaxLot, TaxLotTransaction.tax_lot_id == TaxLot.id
        ).filter(
            Action.portfolio_id == portfolio_id,
            Action.date >= start_date,
            Action.date <= end_date,
            TaxLotTransaction.realized_gain_loss.isnot(None)
        )
        
        if asset_id:
            query = query.filter(TaxLot.asset_id == asset_id)
            
        sell_transactions = query.order_by(Action.date).all()

        realized_data = []
        for trans, action, lot in sell_transactions:
            # Convert to target currency if needed
            if lot.currency_id == currency_id:
                realized_gain_loss = float(trans.realized_gain_loss)
                proceeds = float(trans.proceeds) if trans.proceeds else 0
                cost_basis = proceeds - realized_gain_loss
            else:
                conversion_rate = self.db_access.get_currency_conversion_on_date(
                    session, lot.currency_id, currency_id, action.date
                )
                realized_gain_loss = float(trans.realized_gain_loss) * conversion_rate
                proceeds = float(trans.proceeds) * conversion_rate if trans.proceeds else 0
                cost_basis = proceeds - realized_gain_loss

            # Calculate holding period
            holding_period_days = (action.date.date() - lot.acquisition_date.date()).days
            is_long_term = holding_period_days > 365

            realized_data.append({
                'sale_date': action.date.date(),
                'acquisition_date': lot.acquisition_date.date(),
                'asset_ticker': action.asset.ticker,
                'asset_name': action.asset.name,
                'lot_id': lot.lot_identifier,
                'quantity_sold': abs(float(trans.quantity_change)),
                'sale_price': float(action.price),
                'cost_basis_per_unit': float(lot.cost_basis_per_unit),
                'proceeds': proceeds,
                'cost_basis': cost_basis,
                'realized_gain_loss': realized_gain_loss,
                'holding_period_days': holding_period_days,
                'is_long_term': is_long_term,
                'tax_category': 'Long-term' if is_long_term else 'Short-term'
            })

        return pd.DataFrame(realized_data)

    @with_session
    def get_unrealized_gains_report(self, session: Session, portfolio_id: int, 
                                   asset_id: Optional[int] = None, currency_id: int = 1,
                                   as_of_date: date = datetime.now().date()) -> pd.DataFrame:
        """
        Generate unrealized gains report showing current tax lot positions
        
        Args:
            session: Database session
            portfolio_id: Portfolio ID
            asset_id: Optional asset ID (None for all assets)
            currency_id: Currency to express gains in
            as_of_date: Date to calculate unrealized gains as of
            
        Returns:
            DataFrame with unrealized gain/loss details by lot
        """
        # Get open tax lots
        query = session.query(TaxLot).filter(
            TaxLot.portfolio_id == portfolio_id,
            TaxLot.remaining_quantity > 0,
            TaxLot.is_closed == False
        )
        
        if asset_id:
            query = query.filter(TaxLot.asset_id == asset_id)
            
        open_lots = query.all()

        unrealized_data = []
        for lot in open_lots:
            # Get current market price
            current_price_history = session.query(self.db_access.engine.dialect.name).from_statement(
                f"""
                SELECT close FROM price_history 
                WHERE asset_id = {lot.asset_id} AND date <= '{as_of_date}'
                ORDER BY date DESC LIMIT 1
                """
            ).first()
            
            if not current_price_history:
                continue
                
            current_price = Decimal(str(current_price_history[0]))
            
            # Calculate current market value
            market_value = lot.remaining_quantity * current_price
            
            # Calculate cost basis in target currency
            if lot.currency_id == currency_id:
                cost_basis = lot.remaining_quantity * lot.cost_basis_per_unit
            else:
                conversion_rate = self.db_access.get_currency_conversion_on_date(
                    session, lot.currency_id, currency_id, as_of_date
                )
                cost_basis = lot.remaining_quantity * lot.cost_basis_per_unit * Decimal(str(conversion_rate))
            
            # Convert market value to target currency if needed
            if lot.asset_id != currency_id:
                price_conversion_rate = self.db_access.get_currency_conversion_on_date(
                    session, lot.asset_id, currency_id, as_of_date
                )
                market_value = market_value * Decimal(str(price_conversion_rate))
            
            unrealized_gain_loss = market_value - cost_basis
            
            # Calculate holding period
            holding_period_days = (as_of_date - lot.acquisition_date.date()).days
            is_long_term = holding_period_days > 365

            unrealized_data.append({
                'as_of_date': as_of_date,
                'acquisition_date': lot.acquisition_date.date(),
                'asset_ticker': lot.asset.ticker,
                'asset_name': lot.asset.name,
                'lot_id': lot.lot_identifier,
                'remaining_quantity': float(lot.remaining_quantity),
                'cost_basis_per_unit': float(lot.cost_basis_per_unit),
                'current_price': float(current_price),
                'cost_basis': float(cost_basis),
                'market_value': float(market_value),
                'unrealized_gain_loss': float(unrealized_gain_loss),
                'unrealized_gain_loss_pct': float((unrealized_gain_loss / cost_basis) * 100) if cost_basis > 0 else 0,
                'holding_period_days': holding_period_days,
                'is_long_term': is_long_term,
                'tax_category': 'Long-term' if is_long_term else 'Short-term'
            })

        return pd.DataFrame(unrealized_data)

    @with_session
    def process_action_with_tax_lots(self, session: Session, action: Action, 
                                   method: str = 'fifo') -> None:
        """
        Process any action and update tax lots accordingly
        
        Args:
            session: Database session
            action: Action to process
            method: Tax lot method for sales
        """
        if action.action_type.name == 'buy':
            self.create_tax_lots_from_buy_action(session, action)
        elif action.action_type.name == 'sell':
            self.process_sell_action_with_tax_lots(session, action, method)
        elif action.action_type.name == 'dividend':
            # Dividends don't affect tax lots directly
            logging.info(f"Dividend action {action.id} processed - no tax lot changes")
        else:
            logging.warning(f"Unknown action type {action.action_type.name} for action {action.id}")