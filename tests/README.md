# Database Layer Testing

This directory contains comprehensive tests for the Portfolio Modeling database access layer. The testing suite is designed to ensure reliability, performance, and correctness of all database operations.

## Test Structure

```
tests/
├── conftest.py                 # Pytest fixtures and configuration
├── test_database_access.py     # Unit tests for database access methods
├── integration/
│   └── test_database_integration.py  # Integration tests for workflows
└── README.md                   # This file
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual methods in isolation
- Fast execution (< 1 second per test)
- Mock external dependencies
- Focus on single responsibility

### Integration Tests (`@pytest.mark.integration`)
- Test complete workflows across multiple components
- Test real database interactions
- Verify end-to-end functionality
- May take longer to execute

### Financial Tests (`@pytest.mark.financial`)
- Test financial calculations and precision
- Verify Decimal arithmetic accuracy
- Test currency conversions
- Validate tax lot calculations

### Slow Tests (`@pytest.mark.slow`)
- Performance and bulk operation tests
- Large dataset processing
- Stress testing
- Marked separately to allow fast test runs

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Ensure the main application dependencies are installed:
```bash
pip install -r requirements.txt
```

### Quick Test Commands

```bash
# Run all unit tests (fast)
python run_tests.py unit

# Run integration tests
python run_tests.py integration

# Run all tests with coverage
python run_tests.py coverage

# Run only fast tests (exclude slow tests)
python run_tests.py fast

# Run financial calculation tests
python run_tests.py financial

# Run everything
python run_tests.py all
```

### Advanced Pytest Commands

```bash
# Run specific test file
pytest tests/test_database_access.py -v

# Run specific test class
pytest tests/test_database_access.py::TestAssetManagement -v

# Run specific test method
pytest tests/test_database_access.py::TestAssetManagement::test_add_asset_new -v

# Run tests matching pattern
pytest -k "currency" -v

# Run tests with coverage report
pytest --cov=database --cov-report=html tests/

# Run tests in parallel (requires pytest-xdist)
pytest -n auto tests/

# Stop on first failure
pytest -x tests/

# Show local variables on failure
pytest -l tests/

# Run with detailed output
pytest -vvv tests/
```

## Test Fixtures

### Database Fixtures
- `test_engine`: In-memory SQLite database for testing
- `db_session`: Database session with automatic cleanup
- `db_access`: DatabaseAccess instance with test engine
- `initialized_db`: Database with reference data pre-loaded

### Data Fixtures
- `sample_portfolios`: Test portfolio data
- `sample_assets`: Test asset data including currencies
- `sample_price_history`: Sample price history with Decimal precision
- `high_precision_price_data`: High-precision financial data for testing
- `currency_conversion_test_data`: Currency conversion test scenarios
- `tax_lot_test_data`: Tax lot testing data

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Use descriptive names that explain what is being tested

### Example Test Structure

```python
class TestNewFeature:
    """Test new feature functionality"""

    @pytest.mark.unit
    def test_basic_functionality(self, db_access, db_session):
        """Test basic functionality works correctly"""
        # Arrange
        test_data = {"key": "value"}
        
        # Act
        result = db_access.new_method(db_session, test_data)
        
        # Assert
        assert result is not None
        assert result.key == "value"

    @pytest.mark.integration
    def test_complete_workflow(self, db_access, initialized_db):
        """Test complete workflow integration"""
        # Test multiple operations together
        pass

    @pytest.mark.financial
    def test_financial_precision(self, db_access, db_session):
        """Test financial calculations maintain precision"""
        # Test Decimal precision is maintained
        pass
```

### Best Practices

1. **Use Appropriate Markers**: Mark tests with `@pytest.mark.unit`, `@pytest.mark.integration`, etc.

2. **Test Data Isolation**: Each test should be independent and not rely on other tests

3. **Use Fixtures**: Leverage existing fixtures for common setup

4. **Descriptive Assertions**: Use clear assertion messages
   ```python
   assert result.quantity == expected_quantity, f"Expected {expected_quantity}, got {result.quantity}"
   ```

5. **Test Edge Cases**: Include tests for boundary conditions and error scenarios

6. **Financial Precision**: Always use Decimal for financial calculations in tests
   ```python
   assert result.price == Decimal('123.45')  # Good
   assert result.price == 123.45  # Bad - float comparison
   ```

## Test Data Management

### In-Memory Database
Tests use SQLite in-memory databases (`sqlite:///:memory:`) for:
- Fast test execution
- Isolation between tests
- No cleanup required
- Consistent test environment

### Sample Data
- All sample data uses realistic values
- Financial data uses proper Decimal precision
- Dates use consistent format (datetime objects)
- Foreign key relationships are properly maintained

## Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Critical Paths**: 95% coverage for financial calculations
- **New Code**: 90% coverage required for new features

### Viewing Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=database --cov-report=html tests/

# Open coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Changes to database/ or tests/ directories

### CI Pipeline
1. **Linting**: Code style and syntax checks
2. **Unit Tests**: Fast, isolated tests
3. **Integration Tests**: End-to-end workflows
4. **Financial Tests**: Precision and calculation tests
5. **Coverage Report**: Minimum 75% coverage required
6. **Performance Tests**: Bulk operation benchmarks
7. **Security Scan**: Vulnerability detection

## Debugging Failed Tests

### Common Issues

1. **Decimal vs Float**: Ensure financial values use Decimal
   ```python
   # Wrong
   assert result.price == 123.45
   
   # Correct
   assert result.price == Decimal('123.45')
   ```

2. **Date Comparisons**: Use consistent date types
   ```python
   # Ensure both sides are same type
   assert result.date.date() == expected_date.date()
   ```

3. **Database State**: Tests should not depend on external database state

### Debugging Commands

```bash
# Run single test with detailed output
pytest tests/test_database_access.py::TestAssetManagement::test_add_asset_new -vvv -s

# Run with pdb debugger on failure
pytest --pdb tests/test_database_access.py::TestAssetManagement::test_add_asset_new

# Show local variables on failure
pytest -l tests/test_database_access.py::TestAssetManagement::test_add_asset_new
```

## Performance Testing

Performance tests are marked with `@pytest.mark.slow` and test:
- Bulk data operations
- Large dataset processing
- Query performance
- Memory usage

Run performance tests separately:
```bash
python run_tests.py slow
```

## Contributing

When adding new database functionality:

1. **Write Tests First**: Follow TDD approach
2. **Add Unit Tests**: Test individual methods
3. **Add Integration Tests**: Test complete workflows
4. **Update Fixtures**: Add new test data if needed
5. **Run Full Test Suite**: Ensure no regressions
6. **Check Coverage**: Maintain coverage requirements

## Troubleshooting

### Import Errors
Ensure you're running tests from the project root directory and all dependencies are installed.

### Database Errors
Check that the test database schema matches the entities. Run `db_access.init_db()` in fixtures if needed.

### Fixture Errors
Verify fixture dependencies and scope. Use `pytest --fixtures` to see available fixtures.

### Performance Issues
Use `pytest --durations=10` to identify slow tests and optimize accordingly.
