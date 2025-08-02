import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
from decimal import Decimal
from database.entities_improved import Base, Asset, Portfolio, TransctionType
from database.access_improved import DatabaseAccess
from config import action_types, assets, portfolios


@pytest.fixture(scope="function")
def test_engine():
    """Create in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="function")
def db_session(test_engine):
    """Create a database session for testing"""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="function")
def db_access(test_engine):
    """Create DatabaseAccess instance with test engine"""
    return DatabaseAccess(engine=test_engine)


@pytest.fixture
def sample_portfolios():
    """Sample portfolio data for testing"""
    return [
        {"name": "Test Portfolio 1", "owner": "Test User 1"},
        {"name": "Test Portfolio 2", "owner": "Test User 2"},
        {"name": "Alexander", "owner": "Alexander"}  # Match config data
    ]


@pytest.fixture
def sample_assets():
    """Sample asset data for testing"""
    return [
        {"ticker": "AAPL", "code": "AAPL", "name": "Apple Inc.", "is_currency": False, "is_inverted": False},
        {"ticker": "", "code": "USD", "name": "US Dollar", "is_currency": True, "is_inverted": False},
        {"ticker": "EUR=X", "code": "EUR", "name": "Euro", "is_currency": True, "is_inverted": True},
        {"ticker": "BTC-USD", "code": "BTC", "name": "Bitcoin", "is_currency": True, "is_inverted": False}
    ]


@pytest.fixture
def sample_action_types():
    """Sample action types for testing"""
    return [
        {"name": "buy"},
        {"name": "sell"},
        {"name": "dividend"}
    ]


@pytest.fixture
def sample_price_history():
    """Sample price history data for testing"""
    return [
        {
            'date': datetime(2024, 1, 1),
            'open': Decimal('100.00'),
            'high': Decimal('105.50'),
            'low': Decimal('99.25'),
            'close': Decimal('103.75'),
            'volume': 1000000
        },
        {
            'date': datetime(2024, 1, 2),
            'open': Decimal('103.75'),
            'high': Decimal('107.25'),
            'low': Decimal('102.50'),
            'close': Decimal('106.00'),
            'volume': 1200000
        }
    ]


@pytest.fixture
def high_precision_price_data():
    """High precision price data for testing decimal accuracy"""
    return [
        {
            'date': datetime(2024, 1, 1),
            'open': Decimal('123.456789012345'),
            'high': Decimal('124.567890123456'),
            'low': Decimal('122.345678901234'),
            'close': Decimal('123.987654321098'),
            'volume': 1000000
        }
    ]


@pytest.fixture
def sample_actions():
    """Sample action data for testing"""
    return [
        {
            'portfolio_id': 1,
            'action_type_id': 1,  # buy
            'transaction_datetime': datetime(2024, 1, 1),
            'asset_id': 1,
            'currency_id': 2,  # USD
            'price': Decimal('100.00'),
            'quantity': Decimal('10.0'),
            'fee': Decimal('9.99'),
            'platform': 'Test Broker',
            'comment': 'Test buy action',
            'is_processed': False
        },
        {
            'portfolio_id': 1,
            'action_type_id': 2,  # sell
            'transaction_datetime': datetime(2024, 1, 15),
            'asset_id': 1,
            'currency_id': 2,  # USD
            'price': Decimal('110.00'),
            'quantity': Decimal('5.0'),
            'fee': Decimal('9.99'),
            'platform': 'Test Broker',
            'comment': 'Test sell action',
            'is_processed': False
        }
    ]


@pytest.fixture
def initialized_db(db_access, db_session, sample_portfolios, sample_assets, sample_action_types):
    """Database with basic reference data initialized"""
    # Insert action types
    db_access.insert_if_not_exists(db_session, TransctionType, sample_action_types, ['name'])
    
    # Insert assets
    db_access.insert_if_not_exists(db_session, Asset, sample_assets, ['ticker'])
    
    # Insert portfolios
    db_access.insert_if_not_exists(db_session, Portfolio, sample_portfolios, ['name'])
    
    db_session.commit()
    return db_session


@pytest.fixture
def currency_conversion_test_data():
    """Test data for currency conversion testing"""
    return {
        'usd_price': [
            {
                'date': datetime(2024, 1, 1),
                'open': Decimal('1.0'),
                'high': Decimal('1.0'),
                'low': Decimal('1.0'),
                'close': Decimal('1.0'),
                'volume': 0
            }
        ],
        'eur_price': [
            {
                'date': datetime(2024, 1, 1),
                'open': Decimal('0.85'),
                'high': Decimal('0.86'),
                'low': Decimal('0.84'),
                'close': Decimal('0.85'),
                'volume': 1000000
            }
        ]
    }


@pytest.fixture
def tax_lot_test_data():
    """Test data for tax lot functionality"""
    return {
        'portfolio_id': 1,
        'asset_id': 1,
        'currency_id': 2,
        'lots': [
            {
                'lot_identifier': 'LOT_001',
                'acquisition_date': datetime(2024, 1, 1),
                'original_quantity': Decimal('100.0'),
                'remaining_quantity': Decimal('100.0'),
                'cost_basis_per_unit': Decimal('50.00'),
                'is_closed': False
            },
            {
                'lot_identifier': 'LOT_002',
                'acquisition_date': datetime(2024, 2, 1),
                'original_quantity': Decimal('50.0'),
                'remaining_quantity': Decimal('25.0'),
                'cost_basis_per_unit': Decimal('60.00'),
                'is_closed': False
            }
        ]
    }
