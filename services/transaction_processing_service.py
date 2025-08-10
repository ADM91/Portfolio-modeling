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
from database.entities import Transaction, Portfolio as PortfolioEntity, Asset as AssetEntity
from domain.entities import Portfolio, Asset, TransactionType
from domain.investment_transaction import InvestmentTransaction
from domain.portfolio_holding import PortfolioHolding
from services.excel_import_service import ExcelImportService


class TransactionProcessingService:
    """
    Application service for managing investment transactions.
    Orchestrates domain models, domain services, and infrastructure concerns.
    """
    
    def __init__(self, db_access: DatabaseAccess):
        self.db_access = db_access
        self.excel_import_service = ExcelImportService()
        self.logger = logging.getLogger(__name__)

    @with_session
    def import_transactions_from_excel(self, session: Session, excel_path: str) -> Tuple[int, List[str]]:
        """
        Import transactions from Excel file and save them to the database.

        Args:
            session: Database session
            excel_path: Path to the Excel file
            
        Returns:
            Tuple of (number of actions imported, list of error messages)
        """
        try:
            # Import raw data from Excel
            raw_transactions_data = self.excel_import_service.read_transactions_from_excel(excel_path)
            self.logger.info(f"Read {len(raw_transactions_data)} transactions from Excel")

            if not raw_transactions_data:
                return 0, ["No valid transactions found in Excel file"]

            # Convert to domain models
            transactions, conversion_errors = self._convert_raw_data_to_domain_models(
                session, raw_transactions_data
            )

            if not transactions:
                return 0, conversion_errors or ["No valid transactions could be created from Excel data"]

            # Validate transactions using domain model static method
            validation_errors = PortfolioHolding.validate_transaction_sequence(transactions)
            if validation_errors:
                self.logger.warning(f"Validation errors found: {validation_errors}")
                # Continue with valid transactions, but report errors

            # Save transactions to database
            saved_count = self._save_transactions_to_database(session, transactions)

            all_errors = conversion_errors + validation_errors
            self.logger.info(f"Successfully imported {saved_count} transactions from Excel")

            return saved_count, all_errors
            
        except Exception as e:
            self.logger.error(f"Error importing transactions from Excel: {e}")
            return 0, [f"Import failed: {str(e)}"]

    @with_session
    def process_unprocessed_transactions(self, session: Session) -> Tuple[int, List[str]]:
        """
        Process all unprocessed transactions in the database.

        Args:
            session: Database session
            
        Returns:
            Tuple of (number of transactions processed, list of error messages)
        """
        try:
            # Get unprocessed transactions from database
            unprocessed_entities = self.db_access.get_unprocessed_transactions(session)

            if not unprocessed_entities:
                self.logger.info("No unprocessed transactions found")
                return 0, []

            self.logger.info(f"Found {len(unprocessed_entities)} unprocessed transactions")
            
            # Convert to domain models
            transactions = []
            for entity in unprocessed_entities:
                try:
                    transaction = self._convert_entity_to_domain_model(session, entity)
                    transactions.append(transaction)
                except Exception as e:
                    self.logger.error(f"Error converting transaction entity {entity.id}: {e}")
                    continue

            if not transactions:
                return 0, ["No valid transactions could be processed"]

            # Process transactions using domain model static method
            processed_transactions, processing_errors = PortfolioHolding.process_action_batch(transactions)

            # Update database with processed transactions
            for transaction in processed_transactions:
                try:
                    self._update_transaction_in_database(session, transaction)
                    self._update_holdings_time_series(session, transaction)
                except Exception as e:
                    error_msg = f"Error updating database for transaction {transaction.id}: {e}"
                    processing_errors.append(error_msg)
                    self.logger.error(error_msg)

            self.logger.info(f"Successfully processed {len(processed_transactions)} transactions")
            return len(processed_transactions), processing_errors
            
        except Exception as e:
            self.logger.error(f"Error processing unprocessed transactions: {e}")
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
            # Get all transactions for the portfolio
            transaction_entities = session.query(Transaction).filter(
                Transaction.portfolio_id == portfolio_entity.id
            ).all()
            
            # Convert to domain models
            transactions = []
            for entity in transaction_entities:
                try:
                    transaction = self._convert_entity_to_domain_model(session, entity)
                    transactions.append(transaction)
                except Exception as e:
                    self.logger.warning(f"Skipping transaction {entity.id}: {e}")
                    continue
            
            # Calculate summary using domain model static method
            summary = PortfolioHolding.calculate_portfolio_summary(transactions)

            return summary.get(portfolio_name, {})
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e)}

    def _convert_raw_data_to_domain_models(
        self, 
        session: Session, 
        raw_data: List[Dict]
    ) -> Tuple[List[InvestmentTransaction], List[str]]:
        """
        Convert raw Excel data to domain models.
        
        Args:
            session: Database session
            raw_data: List of raw transaction dictionaries from Excel

        Returns:
            Tuple of (list of InvestmentTransaction objects, list of error messages)
        """
        transactions = []
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
                transaction = InvestmentTransaction(
                    portfolio=portfolio,
                    transaction_type=TransactionType(data['transaction_type']),
                    transaction_datetime=data['transaction_datetime'],
                    asset=asset,
                    currency=currency,
                    price=data['price'],
                    quantity=data['quantity'],
                    fee=data['fee'],
                    platform=data['platform'],
                    comment=data['comment']
                )

                transactions.append(transaction)

            except Exception as e:
                errors.append(f"Row {i}: Error creating transaction - {str(e)}")
                continue

        return transactions, errors

    def _convert_entity_to_domain_model(self, session: Session, entity: Transaction) -> InvestmentTransaction:
        """
        Convert a database entity to a domain model.
        
        Args:
            session: Database session
            entity: TransactionEntity from database

        Returns:
            InvestmentTransaction domain model
        """
        # Get related entities
        portfolio_entity = session.get(PortfolioEntity, entity.portfolio_id)
        asset_entity = session.get(AssetEntity, entity.asset_id)
        currency_entity = session.get(AssetEntity, entity.currency_id)
        
        if not all([portfolio_entity, asset_entity, currency_entity]):
            raise ValueError(f"Missing related entities for transaction {entity.id}")

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
        # Map transaction type
        transaction_type_name = entity.transaction_type.name.lower()
        transaction_type = TransactionType(transaction_type_name)

        # Create domain model
        transaction = InvestmentTransaction(
            portfolio=portfolio,
            transaction_type=transaction_type,
            transaction_datetime=entity.transaction_datetime,
            asset=asset,
            currency=currency,
            price=Decimal(str(entity.price)),
            quantity=Decimal(str(entity.quantity)),
            fee=Decimal(str(entity.fee)),
            platform=entity.platform,
            comment=entity.comment,
            transaction_id=entity.id
        )
        
        transaction.is_processed = entity.is_processed
        return transaction

    def _save_transactions_to_database(self, session: Session, transactions: List[InvestmentTransaction]) -> int:
        """
        Save domain model transactions to the database.
        
        Args:
            session: Database session
            transactions: List of InvestmentTransaction domain models

        Returns:
            Number of transactions successfully saved
        """
        transactions_data = []
        for transaction in transactions:
            transaction_dict = transaction.to_dict()
            # Remove the 'id' key as it will be auto-generated
            transaction_dict.pop('id', None)
            transactions_data.append(transaction_dict)

        # Use existing database method to insert if not exists
        filter_fields = ['portfolio_id', 'action_type_id', 'transaction_datetime', 'asset_id', 'currency_id', 'price', 'quantity']

        # We need to map transaction_type to action_type_id
        for data in transactions_data:
            transaction_type_entity = self.db_access.get_transaction_type_by_name(session, data['transaction_type'])
            if transaction_type_entity:
                data['action_type_id'] = transaction_type_entity.id
                del data['transaction_type']  # Remove the string version

        self.db_access.insert_if_not_exists(session, Transaction, transactions_data, filter_fields)
        return len(transactions_data)

    def _update_transaction_in_database(self, session: Session, transaction: InvestmentTransaction) -> None:
        """
        Update a transaction in the database to mark it as processed.

        Args:
            session: Database session
            transaction: InvestmentAction domain model
        """
        if transaction.id:
            entity = session.get(Transaction, transaction.id)
            if entity:
                entity.is_processed = transaction.is_processed

    def _update_holdings_time_series(self, session: Session, transaction: InvestmentTransaction) -> None:
        """
        Update holdings time series for a transaction.

        Args:
            session: Database session
            transaction: InvestmentAction domain model
        """
        try:
            # Convert domain model back to entity for database operations
            transaction_entity = session.get(Transaction, transaction.id)
            if not transaction_entity:
                self.logger.error(f"Transaction entity {transaction.id} not found for holdings update")
                return
            
            current_date = datetime.now().date()
            
            # Use existing database methods for holdings time series updates
            start_time = time.time()
            self.db_access.insert_holding_time_series_ffill(session, transaction_entity, current_date)
            end_time = time.time()
            self.logger.debug(f"Holdings ffill update took {end_time - start_time:.3f} seconds")
            
            start_time = time.time()
            self.db_access.update_holding_time_series_vectorized(session, transaction_entity, current_date)
            end_time = time.time()
            self.logger.debug(f"Holdings vectorized update took {end_time - start_time:.3f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error updating holdings time series for transaction {transaction.id}: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    db_access = DatabaseAccess()
    transaction_service = TransactionProcessingService(db_access)
    
    # Example: Update holdings time series
    transaction_service.update_holdings_time_series_to_current_day()

    print('TransactionService refactoring complete')
