"""
Security configuration and utilities for Aguas Transparentes API
Implements secure error handling, logging, and request validation
"""

import logging
import json
from datetime import datetime
from typing import Optional
from fastapi import Request, HTTPException, Header, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
import os

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configure security logger
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.INFO)

# File handler for security audit log
security_handler = logging.FileHandler('security_audit.log')
security_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
security_logger.addHandler(security_handler)

# Console handler for development
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)
security_logger.addHandler(console_handler)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors securely
    - Log details internally for debugging
    - Return generic message to client
    """
    security_logger.warning(
        f"Validation error on {request.url} from {request.client.host}: {exc.errors()}"
    )

    # In development, show detailed errors for debugging
    if not IS_PRODUCTION:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Los datos enviados no son válidos. Por favor verifica los parámetros.",
                "errors": exc.errors()  # Only in development
            }
        )

    # In production, hide error details
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Los datos enviados no son válidos. Por favor verifica los parámetros."
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected errors without exposing internal details
    """
    security_logger.error(
        f"Unexpected error on {request.url} from {request.client.host}: {str(exc)}",
        exc_info=True
    )

    # In production, never expose internal error messages
    if IS_PRODUCTION:
        detail = "Ha ocurrido un error interno. Por favor contacta al administrador."
    else:
        detail = f"Error: {str(exc)}"  # Show details in development

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": detail}
    )


# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================

class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests and detect suspicious activity
    """
    async def dispatch(self, request: Request, call_next):
        # Log incoming request
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "x_requested_with": request.headers.get("x-requested-with", "none"),
        }

        # Execute request
        start_time = datetime.utcnow()
        response = await call_next(request)
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Add response data
        log_data.update({
            "status_code": response.status_code,
            "duration_seconds": duration,
        })

        # Log suspicious events
        if response.status_code == 403:
            security_logger.warning(f"Access denied: {json.dumps(log_data)}")
        elif response.status_code == 422:
            security_logger.info(f"Validation error: {json.dumps(log_data)}")
        elif response.status_code >= 500:
            security_logger.error(f"Server error: {json.dumps(log_data)}")

        # Log slow requests (possible DoS)
        if duration > 10.0:
            security_logger.warning(f"Slow request detected: {json.dumps(log_data)}")

        return response


class LimitPayloadSizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request payload size
    Prevents memory exhaustion attacks
    """
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10 MB default
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        # Check Content-Length header
        content_length = request.headers.get("content-length")

        if content_length:
            content_length = int(content_length)
            if content_length > self.max_size:
                security_logger.warning(
                    f"Payload too large from {request.client.host if request.client else 'unknown'}: "
                    f"{content_length} bytes (max: {self.max_size})"
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Payload demasiado grande. Máximo permitido: {self.max_size / 1024 / 1024}MB"
                    }
                )

        response = await call_next(request)
        return response


# ============================================================================
# DEPENDENCY FUNCTIONS
# ============================================================================

async def validate_xhr_header(
    x_requested_with: Optional[str] = Header(None)
):
    """
    Validate X-Requested-With header for CSRF protection
    The frontend should send: X-Requested-With: XMLHttpRequest

    This can be disabled in development by setting REQUIRE_XHR_HEADER=false
    """
    require_xhr = os.getenv("REQUIRE_XHR_HEADER", "false").lower() == "true"

    if require_xhr and x_requested_with != "XMLHttpRequest":
        security_logger.warning(f"Request without valid X-Requested-With header")
        raise HTTPException(
            status_code=403,
            detail="Petición no autorizada"
        )


# ============================================================================
# LOGGING UTILITY FUNCTIONS
# ============================================================================

def log_failed_validation(endpoint: str, ip: str, errors: list):
    """Log failed validation attempts (possible injection attacks)"""
    security_logger.warning(
        f"Failed validation - Endpoint: {endpoint}, IP: {ip}, Errors: {errors}"
    )


def log_suspicious_activity(ip: str, reason: str, details: dict = None):
    """Log suspicious activity for security monitoring"""
    log_msg = f"SUSPICIOUS ACTIVITY - IP: {ip}, Reason: {reason}"
    if details:
        log_msg += f", Details: {json.dumps(details)}"
    security_logger.error(log_msg)


def log_database_error(operation: str, error: Exception, query: str = None):
    """Log database errors securely (without exposing sensitive data)"""
    security_logger.error(
        f"Database error during {operation}: {str(error)}"
        # Do NOT log the full query as it may contain sensitive data
    )
