import azure.functions as func
import logging
import json
from urllib.parse import urlparse, parse_qs

# Import FastAPI app and dependencies
from main import app as fastapi_app
from fastapi.testclient import TestClient

# Create Azure Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Create test client for FastAPI
client = TestClient(fastapi_app)

@app.function_name(name="main")
@app.route(route="{*route}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
async def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main HTTP trigger that routes all requests to FastAPI"""
    logging.info(f'Processing request: {req.method} {req.url}')

    # Test basic functionality first
    if req.url.endswith("/test"):
        return func.HttpResponse(
            '{"message": "Azure Functions working!"}',
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
        # These need to be accessed without /api prefix
        if path in ['', '/']:
            path = '/'
        elif path.startswith('/docs') or path.startswith('/redoc') or path.startswith('/openapi.json'):
            # These paths are already correct, no modification needed
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