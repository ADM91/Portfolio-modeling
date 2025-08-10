"""
Infrastructure service for importing data from Excel files.
This service handles data transformation and validation but not business logic.
"""

import logging
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal


class ExcelImportService:
    """
    Service for importing investment action data from Excel files.
    Handles data parsing, transformation, and basic validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def read_transactions_from_excel(self, excel_path: str) -> List[Dict]:
        """
        Read investment transactions from an Excel file and return raw data.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            List of dictionaries containing raw action data
            
        Raises:
            FileNotFoundError: If the Excel file doesn't exist
            ValueError: If the Excel file format is invalid
        """
        try:
            transaction_df = pd.read_excel(excel_path)
            self.logger.info(f"Successfully read {len(transaction_df)} rows from {excel_path}")
        except FileNotFoundError:
            self.logger.error(f"Excel file not found: {excel_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to read Excel file {excel_path}: {e}")
            raise ValueError(f"Invalid Excel file format: {e}")
        
        # Validate required columns
        required_columns = ['Datetime', 'Portfolio', 'Transaction', 'Asset', 'Currency', 'Price', 'Quantity']
        missing_columns = [col for col in required_columns if col not in transaction_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        transaction_list = []

        for index, row in transaction_df.iterrows():
            try:
                processed_row = self._process_excel_row(row, index)
                if processed_row:  # Skip rows that couldn't be processed
                    transaction_list.append(processed_row)
            except Exception as e:
                self.logger.error(f"Error processing row {index}: {e}")
                self.logger.error(f"Row data: {row.to_dict()}")
                # Continue processing other rows instead of failing completely
                continue
        
        self.logger.info(f"Successfully processed {len(transaction_list)} transactions from Excel")
        return transaction_list

    def _process_excel_row(self, row: pd.Series, row_index: int) -> Optional[Dict]:
        """
        Process a single Excel row into a standardized dictionary.
        
        Args:
            row: Pandas Series representing a row from the Excel file
            row_index: Index of the row for error reporting
            
        Returns:
            Dictionary with processed data, or None if row should be skipped
        """
        try:
            # Parse and validate date
            datetime_value = self._parse_date(row['Datetime'])
            if datetime_value is None:
                self.logger.warning(f"Skipping row {row_index}: Invalid date '{row['Datetime']}'")
                return None
            
            # Parse numeric values
            price = self._parse_decimal(row['Price'], 'Price', row_index)
            quantity = self._parse_decimal(row['Quantity'], 'Quantity', row_index)
            fee = self._parse_decimal(row.get('Fee', 0), 'Fee', row_index)
            
            if price is None or quantity is None:
                return None  # Skip row if critical numeric values are invalid

            # Validate transaction type
            transaction_type = str(row['Transaction']).lower().strip()
            if transaction_type not in ['buy', 'sell', 'dividend']:
                self.logger.warning(f"Skipping row {row_index}: Invalid transaction type '{transaction_type}'")
                return None
            
            # Build processed row
            processed_row = {
                'transaction_datetime': datetime_value,
                'portfolio_name': str(row['Portfolio']).strip(),
                'transaction_type': transaction_type,
                'asset_code': str(row['Asset']).strip(),
                'currency_code': str(row['Currency']).strip(),
                'price': price,
                'quantity': quantity,
                'fee': fee,
                'platform': str(row.get('Platform', '')).strip() or None,
                'comment': str(row.get('Comment', '')).strip() or None,
                'is_processed': False
            }
            
            return processed_row
            
        except Exception as e:
            self.logger.error(f"Error processing row {row_index}: {e}")
            return None

    def _parse_date(self, date_value) -> Optional[datetime]:
        """
        Parse a date value from Excel into a datetime object.
        
        Args:
            date_value: Date value from Excel (could be string, datetime, etc.)
            
        Returns:
            Parsed datetime object, or None if parsing fails
        """
        try:
            if pd.isna(date_value):
                return None
            
            # If it's already a datetime, convert to date and back to datetime
            if isinstance(date_value, datetime):
                return pd.to_datetime(date_value.date())
            
            # Try to parse as datetime with flexible format
            parsed_date = pd.to_datetime(date_value, dayfirst=True, format='mixed', errors='coerce')
            
            if pd.isna(parsed_date):
                return None
            
            # Convert to date and back to datetime to normalize time component
            return pd.to_datetime(parsed_date.date())
            
        except Exception:
            return None

    def _parse_decimal(self, value, field_name: str, row_index: int) -> Optional[Decimal]:
        """
        Parse a numeric value into a Decimal for financial precision.
        
        Args:
            value: Numeric value to parse
            field_name: Name of the field for error reporting
            row_index: Row index for error reporting
            
        Returns:
            Parsed Decimal value, or None if parsing fails
        """
        try:
            if pd.isna(value):
                if field_name in ['Price', 'Quantity']:
                    self.logger.warning(f"Row {row_index}: Missing required field '{field_name}'")
                    return None
                else:
                    return Decimal('0')  # Default for optional fields like Fee
            
            # Convert to string first to handle various numeric types
            decimal_value = Decimal(str(value))
            
            # Validate that critical fields are positive
            if field_name in ['Price', 'Quantity'] and decimal_value <= 0:
                self.logger.warning(f"Row {row_index}: {field_name} must be positive, got {decimal_value}")
                return None
            
            # Validate that fee is non-negative
            if field_name == 'Fee' and decimal_value < 0:
                self.logger.warning(f"Row {row_index}: Fee cannot be negative, got {decimal_value}")
                return Decimal('0')
            
            return decimal_value
            
        except (ValueError, TypeError, pd.errors.ParserError) as e:
            self.logger.warning(f"Row {row_index}: Invalid {field_name} value '{value}': {e}")
            return None

    def validate_excel_format(self, excel_path: str) -> List[str]:
        """
        Validate the format of an Excel file without fully processing it.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            # Try to read the file
            df = pd.read_excel(excel_path)
            
            # Check for required columns
            required_columns = ['Datetime', 'Portfolio', 'Transaction', 'Asset', 'Currency', 'Price', 'Quantity']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check if file is empty
            if len(df) == 0:
                errors.append("Excel file is empty")
            
            # Check for completely empty required columns
            for col in required_columns:
                if col in df.columns and df[col].isna().all():
                    errors.append(f"Column '{col}' is completely empty")
            
        except FileNotFoundError:
            errors.append(f"Excel file not found: {excel_path}")
        except Exception as e:
            errors.append(f"Error reading Excel file: {e}")
        
        return errors

    def get_excel_summary(self, excel_path: str) -> Dict:
        """
        Get a summary of the Excel file contents without full processing.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Dictionary with summary information
        """
        try:
            df = pd.read_excel(excel_path)
            
            summary = {
                'total_rows': len(df),
                'columns': list(df.columns),
                'portfolios': df['Portfolio'].unique().tolist() if 'Portfolio' in df.columns else [],
                'transaction_types': df['Transaction'].unique().tolist() if 'Transaction' in df.columns else [],
                'assets': df['Asset'].unique().tolist() if 'Asset' in df.columns else [],
                'datetime_range': {
                    'start': df['Datetime'].min() if 'Datetime' in df.columns else None,
                    'end': df['Datetime'].max() if 'Datetime' in df.columns else None
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting Excel summary: {e}")
            return {'error': str(e)}
