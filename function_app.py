import azure.functions as func
import logging
import json
import os
from urllib.parse import urlparse, parse_qs

# Import FastAPI app and dependencies
from main import app as fastapi_app
from fastapi.testclient import TestClient

# Create Azure Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Determine if running in Azure or local
IS_AZURE = bool(os.getenv('WEBSITE_INSTANCE_ID'))

# Create test client for local development only
if not IS_AZURE:
    client = TestClient(fastapi_app)
    logging.info("Running in LOCAL mode - using TestClient")
else:
    logging.info("Running in AZURE mode - using AsgiMiddleware")

@app.function_name(name="main")
@app.route(route="{*route}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"])
async def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main HTTP trigger that routes all requests to FastAPI"""
    logging.info(f'Processing request: {req.method} {req.url}')

    # Use ASGI middleware in Azure (production)
    if IS_AZURE:
        try:
            return await func.AsgiMiddleware(fastapi_app).handle_async(req)
        except Exception as e:
            logging.error(f'Error in AsgiMiddleware: {str(e)}')
            import traceback
            logging.error(f'Traceback: {traceback.format_exc()}')
            return func.HttpResponse(
                json.dumps({"error": str(e)}),
                status_code=500,
                mimetype="application/json"
            )

    # Use TestClient in local development
    else:
        # Test basic functionality first
        if req.url.endswith("/test"):
            return func.HttpResponse(
                '{"message": "Azure Functions working in LOCAL mode!"}',
                status_code=200,
                mimetype="application/json"
            )

        try:
            # Extract path from the URL (remove /api prefix)
            parsed_url = urlparse(req.url)
            path = parsed_url.path

            # Handle /api prefix removal, but preserve docs and openapi.json paths
            if path.startswith('/api'):
                path = path[4:]  # Remove /api prefix

            # Special handling for FastAPI docs endpoints
            if path in ['', '/']:
                path = '/'
            elif path.startswith('/docs') or path.startswith('/redoc') or path.startswith('/openapi.json'):
                pass

            # Get query parameters
            query_params = parse_qs(parsed_url.query)
            flat_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}

            # Get request body
            try:
                body = req.get_body().decode('utf-8') if req.get_body() else ""
            except:
                body = ""

            # Get headers
            headers = dict(req.headers)

            logging.info(f'Calling FastAPI path: {path} with method: {req.method}')

            # Call FastAPI using TestClient
            response = client.request(
                method=req.method,
                url=path,
                params=flat_params,
                data=body if body else None,
                headers=headers
            )

            logging.info(f'FastAPI response status: {response.status_code}')

            return func.HttpResponse(
                response.content,
                status_code=response.status_code,
                mimetype=response.headers.get('content-type', 'application/json'),
                headers=dict(response.headers)
            )

        except Exception as e:
            logging.error(f'Error processing request: {str(e)}')
            import traceback
            logging.error(f'Traceback: {traceback.format_exc()}')
            return func.HttpResponse(
                json.dumps({"error": str(e)}),
                status_code=500,
                mimetype="application/json"
            )