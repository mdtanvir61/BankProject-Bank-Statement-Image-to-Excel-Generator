from flask import Flask
from flask.cli import ScriptInfo
import os
import json
from app import app as flask_app

def handler(event, context):
    # Create a new ScriptInfo object
    script_info = ScriptInfo(create_app=lambda info: flask_app)

    # Get the Flask application
    app = script_info.load_app()

    # Parse the event
    path = event['path']
    http_method = event['httpMethod']
    headers = event['headers']
    body = event.get('body', '')
    
    # Convert API Gateway format to WSGI format
    environ = {
        'REQUEST_METHOD': http_method,
        'SCRIPT_NAME': '',
        'PATH_INFO': path,
        'QUERY_STRING': event.get('queryStringParameters', ''),
        'SERVER_NAME': 'netlify',
        'SERVER_PORT': '443',
        'SERVER_PROTOCOL': 'HTTP/1.1',
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': 'https',
        'wsgi.input': body,
        'wsgi.errors': '',
        'wsgi.multithread': False,
        'wsgi.multiprocess': False,
        'wsgi.run_once': False,
    }

    # Add headers
    for key, value in headers.items():
        key = key.upper().replace('-', '_')
        if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
            key = f'HTTP_{key}'
        environ[key] = value

    # Run the request
    response = app.wsgi_app(environ, lambda status, headers: None)

    # Format the response
    return {
        'statusCode': response.status_code,
        'headers': dict(response.headers),
        'body': response.get_data(as_text=True)
    }
