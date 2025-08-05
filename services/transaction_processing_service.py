"""
Refactored ActionService using domain models and clean architecture principles.
This service orchestrates business operations and coordinates between domain models,
domain services, and infrastructure services.
"""

import logging
import time
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session

from database.access import with_session, DatabaseAccess
from database.entities import Action as ActionEntity, Portfolio as PortfolioEntity, Asset as AssetEntity
from domain import InvestmentAction, Portfolio, Asset, ActionType, PortfolioHolding
from services.excel_import_service import ExcelImportService


class TransactionProcessingService:
    """
    Application service for managing investment tranactions.
    Orchestrates domain models, domain services, and infrastructure concerns.
    """
    
    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access
        self.excel_import_service = ExcelImportService()
        self.logger = logging.getLogger(__name__)

    @with_session
    def import_actions_from_excel(self, session: Session, excel_path: str) -> Tuple[int, List[str]]:
        """
        Import actions from Excel file and save them to the database.
        
        Args:
            session: Database session
            excel_path: Path to the Excel file
            
        Returns:
            Tuple of (number of actions imported, list of error messages)
        """
        try:
            # Import raw data from Excel
            raw_actions_data = self.excel_import_service.read_actions_from_excel(excel_path)
            self.logger.info(f"Read {len(raw_actions_data)} actions from Excel")
            
            if not raw_actions_data:
                return 0, ["No valid actions found in Excel file"]
            
            # Convert to domain models
            actions, conversion_errors = self._convert_raw_data_to_domain_models(
                session, raw_actions_data
            )
            
            if not actions:
                return 0, conversion_errors or ["No valid actions could be created from Excel data"]
            
            # Validate actions using domain model static method
            validation_errors = PortfolioHolding.validate_action_sequence(actions)
            if validation_errors:
                self.logger.warning(f"Validation errors found: {validation_errors}")
                # Continue with valid actions, but report errors
            
            # Save actions to database
            saved_count = self._save_actions_to_database(session, actions)
            
            all_errors = conversion_errors + validation_errors
            self.logger.info(f"Successfully imported {saved_count} actions from Excel")
            
            return saved_count, all_errors
            
        except Exception as e:
            self.logger.error(f"Error importing actions from Excel: {e}")
            return 0, [f"Import failed: {str(e)}"]

    @with_session
    def process_unprocessed_actions(self, session: Session) -> Tuple[int, List[str]]:
        """
        Process all unprocessed actions in the database.
        
        Args:
            session: Database session
            
        Returns:
            Tuple of (number of actions processed, list of error messages)
        """
        try:
            # Get unprocessed actions from database
            unprocessed_entities = self.db_access.get_unprocessed_actions(session)
            
            if not unprocessed_entities:
                self.logger.info("No unprocessed actions found")
                return 0, []
            
            self.logger.info(f"Found {len(unprocessed_entities)} unprocessed actions")
            
            # Convert to domain models
            actions = []
            for entity in unprocessed_entities:
                try:
                    action = self._convert_entity_to_domain_model(session, entity)
                    actions.append(action)
                except Exception as e:
                    self.logger.error(f"Error converting action entity {entity.id}: {e}")
                    continue
            
            if not actions:
                return 0, ["No valid actions could be processed"]
            
            # Process actions using domain model static method
            processed_actions, processing_errors = PortfolioHolding.process_action_batch(actions)
            
            # Update database with processed actions
            for action in processed_actions:
                try:
                    self._update_action_in_database(session, action)
                    self._update_holdings_time_series(session, action)
                except Exception as e:
                    error_msg = f"Error updating database for action {action.id}: {e}"
                    processing_errors.append(error_msg)
                    self.logger.error(error_msg)
            
            self.logger.info(f"Successfully processed {len(processed_actions)} actions")
            return len(processed_actions), processing_errors
            
        except Exception as e:
            self.logger.error(f"Error processing unprocessed actions: {e}")
            return 0, [f"Processing failed: {str(e)}"]

    @with_session
    def update_holdings_time_series_to_current_day(self, session: Session) -> None:
        """
        Update holdings time series to the current day for all portfolios.
        
        Args:
            session: Database session
        """
        try:
            self.logger.info("Updating holdings time series to current day")
            start_time = time.time()
            
            self.db_access.update_holdings_time_series_to_current_day(session)
            
            end_time = time.time()
            runtime = end_time - start_time
            self.logger.info(f"Holdings time series update completed in {runtime:.3f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error updating holdings time series: {e}")
            raise

    @with_session
    def get_portfolio_summary(self, session: Session, portfolio_name: str) -> Dict:
        """
        Get a summary of portfolio positions and performance.
        
        Args:
            session: Database session
            portfolio_name: Name of the portfolio
            
        Returns:
            Dictionary with portfolio summary information
        """
        try:
            # Get portfolio
            portfolio_entity = self.db_access.get_portfolio_by_name(session, portfolio_name)
            if not portfolio_entity:
                return {'error': f'Portfolio "{portfolio_name}" not found'}
            
            # Get all actions for the portfolio
            action_entities = session.query(ActionEntity).filter(
                ActionEntity.portfolio_id == portfolio_entity.id
            ).all()
            
            # Convert to domain models
            actions = []
            for entity in action_entities:
                try:
                    action = self._convert_entity_to_domain_model(session, entity)
                    actions.append(action)
                except Exception as e:
                    self.logger.warning(f"Skipping action {entity.id}: {e}")
                    continue
            
            # Calculate summary using domain model static method
            summary = PortfolioHolding.calculate_portfolio_summary(actions)
            
            return summary.get(portfolio_name, {})
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e)}

    def _convert_raw_data_to_domain_models(
        self, 
        session: Session, 
        raw_data: List[Dict]
    ) -> Tuple[List[InvestmentAction], List[str]]:
        """
        Convert raw Excel data to domain models.
        
        Args:
            session: Database session
            raw_data: List of raw action dictionaries from Excel
            
        Returns:
            Tuple of (list of InvestmentAction objects, list of error messages)
        """
        actions = []
        errors = []
        
        for i, data in enumerate(raw_data):
            try:
                # Look up related entities
                portfolio_entity = self.db_access.get_portfolio_by_name(session, data['portfolio_name'])
                if not portfolio_entity:
                    errors.append(f"Row {i}: Portfolio '{data['portfolio_name']}' not found")
                    continue
                
                asset_entity = self.db_access.get_asset_by_code(session, data['asset_code'])
                if not asset_entity:
                    errors.append(f"Row {i}: Asset '{data['asset_code']}' not found")
                    continue
                
                currency_entity = self.db_access.get_asset_by_code(session, data['currency_code'])
                if not currency_entity:
                    errors.append(f"Row {i}: Currency '{data['currency_code']}' not found")
                    continue
                
                # Convert to domain models
                portfolio = Portfolio(
                    id=portfolio_entity.id,
                    name=portfolio_entity.name,
                    owner=portfolio_entity.owner
                )
                
                asset = Asset(
                    id=asset_entity.id,
                    ticker=asset_entity.ticker,
                    code=asset_entity.code,
                    name=asset_entity.name,
                    is_currency=asset_entity.is_currency,
                    is_inverted=asset_entity.is_inverted
                )
                
                currency = Asset(
                    id=currency_entity.id,
                    ticker=currency_entity.ticker,
                    code=currency_entity.code,
                    name=currency_entity.name,
                    is_currency=currency_entity.is_currency,
                    is_inverted=currency_entity.is_inverted
                )
                
                # Create domain model
                action = InvestmentAction(
                    portfolio=portfolio,
                    action_type=ActionType(data['action_type']),
                    date=data['date'],
                    asset=asset,
                    currency=currency,
                    price=data['price'],
                    quantity=data['quantity'],
                    fee=data['fee'],
                    platform=data['platform'],
                    comment=data['comment']
                )
                
                actions.append(action)
                
            except Exception as e:
                errors.append(f"Row {i}: Error creating action - {str(e)}")
                continue
        
        return actions, errors

    def _convert_entity_to_domain_model(self, session: Session, entity: ActionEntity) -> InvestmentAction:
        """
        Convert a database entity to a domain model.
        
        Args:
            session: Database session
            entity: ActionEntity from database
            
        Returns:
            InvestmentAction domain model
        """
        # Get related entities
        portfolio_entity = session.get(PortfolioEntity, entity.portfolio_id)
        asset_entity = session.get(AssetEntity, entity.asset_id)
        currency_entity = session.get(AssetEntity, entity.currency_id)
        
        if not all([portfolio_entity, asset_entity, currency_entity]):
            raise ValueError(f"Missing related entities for action {entity.id}")
        
        # Convert to domain models
        portfolio = Portfolio(
            id=portfolio_entity.id,
            name=portfolio_entity.name,
            owner=portfolio_entity.owner
        )
        
        asset = Asset(
            id=asset_entity.id,
            ticker=asset_entity.ticker,
            code=asset_entity.code,
            name=asset_entity.name,
            is_currency=asset_entity.is_currency,
            is_inverted=asset_entity.is_inverted
        )
        
        currency = Asset(
            id=currency_entity.id,
            ticker=currency_entity.ticker,
            code=currency_entity.code,
            name=currency_entity.name,
            is_currency=currency_entity.is_currency,
            is_inverted=currency_entity.is_inverted
        )
        
        # Map action type
        action_type_name = entity.action_type.name.lower()
        action_type = ActionType(action_type_name)
        
        # Create domain model
        action = InvestmentAction(
            portfolio=portfolio,
            action_type=action_type,
            date=entity.date,
            asset=asset,
            currency=currency,
            price=Decimal(str(entity.price)),
            quantity=Decimal(str(entity.quantity)),
            fee=Decimal(str(entity.fee)),
            platform=entity.platform,
            comment=entity.comment,
            action_id=entity.id
        )
        
        action.is_processed = entity.is_processed
        return action

    def _save_actions_to_database(self, session: Session, actions: List[InvestmentAction]) -> int:
        """
        Save domain model actions to the database.
        
        Args:
            session: Database session
            actions: List of InvestmentAction domain models
            
        Returns:
            Number of actions successfully saved
        """
        actions_data = []
        for action in actions:
            action_dict = action.to_dict()
            # Remove the 'id' key as it will be auto-generated
            action_dict.pop('id', None)
            actions_data.append(action_dict)
        
        # Use existing database method to insert if not exists
        filter_fields = ['portfolio_id', 'action_type_id', 'date', 'asset_id', 'currency_id', 'price', 'quantity']
        
        # We need to map action_type to action_type_id
        for data in actions_data:
            action_type_entity = self.db_access.get_action_type_by_name(session, data['action_type'])
            if action_type_entity:
                data['action_type_id'] = action_type_entity.id
                del data['action_type']  # Remove the string version
        
        self.db_access.insert_if_not_exists(session, ActionEntity, actions_data, filter_fields)
        return len(actions_data)

    def _update_action_in_database(self, session: Session, action: InvestmentAction) -> None:
        """
        Update an action in the database to mark it as processed.
        
        Args:
            session: Database session
            action: InvestmentAction domain model
        """
        if action.id:
            entity = session.get(ActionEntity, action.id)
            if entity:
                entity.is_processed = action.is_processed

    def _update_holdings_time_series(self, session: Session, action: InvestmentAction) -> None:
        """
        Update holdings time series for an action.
        
        Args:
            session: Database session
            action: InvestmentAction domain model
        """
        try:
            # Convert domain model back to entity for database operations
            action_entity = session.get(ActionEntity, action.id)
            if not action_entity:
                self.logger.error(f"Action entity {action.id} not found for holdings update")
                return
            
            current_date = datetime.now().date()
            
            # Use existing database methods for holdings time series updates
            start_time = time.time()
            self.db_access.insert_holding_time_series_ffill(session, action_entity, current_date)
            end_time = time.time()
            self.logger.debug(f"Holdings ffill update took {end_time - start_time:.3f} seconds")
            
            start_time = time.time()
            self.db_access.update_holding_time_series_vectorized(session, action_entity, current_date)
            end_time = time.time()
            self.logger.debug(f"Holdings vectorized update took {end_time - start_time:.3f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error updating holdings time series for action {action.id}: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    db_access = DatabaseAccess()
    action_service = TransactionProcessingService(db_access)
    
    # Example: Import actions from Excel
    # imported_count, errors = action_service.import_actions_from_excel("path/to/actions.xlsx")
    # print(f"Imported {imported_count} actions")
    # if errors:
    #     print("Errors:", errors)
    
    # Example: Process unprocessed actions
    # processed_count, errors = action_service.process_unprocessed_actions()
    # print(f"Processed {processed_count} actions")
    
    # Example: Update holdings time series
    action_service.update_holdings_time_series_to_current_day()
    
    print('ActionService refactoring complete')
