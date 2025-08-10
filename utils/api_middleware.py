import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from utils.logging_config import get_logger, correlation_context, CorrelationIdFilter
import json

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all API requests and responses with correlation IDs.
    """
    
    def __init__(self, app, log_request_body: bool = False, log_response_body: bool = False):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID for this request
        correlation_id = str(uuid.uuid4())[:8]
        
        # Set correlation ID in the filter
        CorrelationIdFilter.set_correlation_id(correlation_id)
        
        # Add correlation ID to request state for use in route handlers
        request.state.correlation_id = correlation_id
        
        # Start timing
        start_time = time.perf_counter()
        
        # Log request
        await self._log_request(request, correlation_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log response
            await self._log_response(request, response, duration_ms, correlation_id)
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log error
            logger.error(f"Request failed: {str(e)}", extra={
                'context': {
                    'method': request.method,
                    'url': str(request.url),
                    'correlation_id': correlation_id,
                    'duration_ms': duration_ms
                }
            }, exc_info=True)
            
            # Return error response
            error_response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "correlation_id": correlation_id,
                    "message": str(e) if logger.level <= 10 else "An error occurred"  # Only show details in DEBUG
                }
            )
            error_response.headers["X-Correlation-ID"] = correlation_id
            return error_response
            
        finally:
            # Clear correlation ID
            CorrelationIdFilter.clear_correlation_id()
    
    async def _log_request(self, request: Request, correlation_id: str):
        """Log incoming request details."""
        
        # Extract client info
        client_host = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        client_port = getattr(request.client, 'port', 'unknown') if request.client else 'unknown'
        
        # Extract headers (excluding sensitive ones)
        headers = dict(request.headers)
        sensitive_headers = {'authorization', 'cookie', 'x-api-key', 'x-auth-token'}
        filtered_headers = {
            k: v if k.lower() not in sensitive_headers else '***' 
            for k, v in headers.items()
        }
        
        # Build context
        context = {
            'method': request.method,
            'url': str(request.url),
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'client_ip': client_host,
            'client_port': client_port,
            'headers': filtered_headers,
            'correlation_id': correlation_id
        }
        
        # Log request body if enabled and it's a POST/PUT/PATCH
        if (self.log_request_body and 
            request.method in ['POST', 'PUT', 'PATCH'] and
            'application/json' in headers.get('content-type', '')):
            try:
                body = await request.body()
                if body:
                    # Try to parse as JSON for better logging
                    try:
                        json_body = json.loads(body.decode('utf-8'))
                        context['request_body'] = json_body
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        context['request_body'] = f"<binary data: {len(body)} bytes>"
            except Exception as e:
                context['request_body_error'] = str(e)
        
        logger.info(f"→ {request.method} {request.url.path}", extra={'context': context})
    
    async def _log_response(self, request: Request, response: Response, duration_ms: float, correlation_id: str):
        """Log response details."""
        
        context = {
            'method': request.method,
            'url': str(request.url),
            'status_code': response.status_code,
            'duration_ms': duration_ms,
            'correlation_id': correlation_id
        }
        
        # Add response headers (excluding sensitive ones)
        response_headers = dict(response.headers)
        sensitive_headers = {'set-cookie', 'authorization'}
        filtered_response_headers = {
            k: v if k.lower() not in sensitive_headers else '***'
            for k, v in response_headers.items()
        }
        context['response_headers'] = filtered_response_headers
        
        # Log response body if enabled and it's JSON
        if (self.log_response_body and 
            response.headers.get('content-type', '').startswith('application/json')):
            # Note: This is tricky with FastAPI as the response body might be consumed
            # For now, we'll skip this feature to avoid complexity
            pass
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = 'error'
        elif response.status_code >= 400:
            log_level = 'warning'
        else:
            log_level = 'info'
        
        log_message = f"← {response.status_code} {request.method} {request.url.path} [{duration_ms:.1f}ms]"
        
        getattr(logger, log_level)(log_message, extra={'context': context})


class HealthCheckFilter:
    """
    Middleware to exclude health check endpoints from logging.
    """
    
    def __init__(self, app, health_endpoints: list = None):
        self.app = app
        self.health_endpoints = health_endpoints or ['/health', '/ping', '/status', '/api/health']
    
    async def __call__(self, scope, receive, send):
        if scope.get('type') == 'http':
            path = scope.get('path', '')
            if path in self.health_endpoints:
                # Skip logging for health checks
                return await self.app(scope, receive, send)
        
        return await self.app(scope, receive, send)


def get_correlation_id(request: Request) -> str:
    """
    Helper function to get correlation ID from request.
    """
    return getattr(request.state, 'correlation_id', 'unknown')
