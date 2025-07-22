# Authentication

The Steel Defect Prediction System uses a robust authentication and authorization system to ensure secure access to
APIs and protect sensitive operational data.

## Overview

Authentication features include:

- JWT-based token authentication
- Role-based access control (RBAC)
- API key authentication for service integrations
- OAuth 2.0 support for third-party integrations
- Multi-factor authentication (MFA) for sensitive operations

## Authentication Methods

### 1. JWT Token Authentication

#### Login and Token Generation

```python
import requests

# User login

login_response = requests.post(
    'http://localhost:8000/api/v1/auth/login',
    json={
        'username': 'operator_001',
        'password': 'secure_password'
    }
)

if login_response.status_code == 200:
    tokens = login_response.json()
    access_token = tokens['access_token']
    refresh_token = tokens['refresh_token']
    
    print(f"Access token expires in: {tokens['expires_in']} seconds")
```text

#### Using Access Tokens

```python

# Make authenticated API calls

headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

# Get current user info

user_info = requests.get(
    'http://localhost:8000/api/v1/auth/me',
    headers=headers
)

print(f"Current user: {user_info.json()['username']}")
print(f"Role: {user_info.json()['role']}")
```text

#### Token Refresh

```python

# Refresh expired access token

refresh_response = requests.post(
    'http://localhost:8000/api/v1/auth/refresh',
    json={'refresh_token': refresh_token}
)

if refresh_response.status_code == 200:
    new_tokens = refresh_response.json()
    access_token = new_tokens['access_token']
```text

### 2. API Key Authentication

#### Generate API Key

```python

# Generate API key for service integration

api_key_response = requests.post(
    'http://localhost:8000/api/v1/auth/api-keys',
    headers=headers,
    json={
        'name': 'SCADA Integration',
        'description': 'API key for SCADA system integration',
        'permissions': ['predictions:read', 'sensors:read'],
        'expires_at': '2024-12-31T23:59:59Z'
    }
)

api_key = api_key_response.json()['api_key']
```text

#### Using API Key

```python

# Use API key for authentication

api_headers = {
    'X-API-Key': api_key,
    'Content-Type': 'application/json'
}

# Make API call with API key

prediction_response = requests.post(
    'http://localhost:8000/api/v1/predictions',
    headers=api_headers,
    json=sensor_data
)
```text

### 3. OAuth 2.0 Integration

#### Authorization Code Flow

```python

# Redirect user to authorization endpoint

auth_url = (
    'http://localhost:8000/api/v1/oauth/authorize'
    '?client_id=your_client_id'
    '&response_type=code'
    '&redirect_uri=http://your-app.com/callback'
    '&scope=predictions:read sensors:write'
    '&state=random_state_string'
)

# User authorizes and is redirected with code

# Exchange code for tokens

token_response = requests.post(
    'http://localhost:8000/api/v1/oauth/token',
    data={
        'grant_type': 'authorization_code',
        'code': 'authorization_code_from_callback',
        'redirect_uri': 'http://your-app.com/callback',
        'client_id': 'your_client_id',
        'client_secret': 'your_client_secret'
    }
)

oauth_tokens = token_response.json()
```text

## User Roles and Permissions

### Role Definitions

```python

# Role hierarchy and permissions

roles = {
    'operator': {
        'permissions': [
            'dashboard:read',
            'predictions:read',
            'alerts:read',
            'alerts:acknowledge'
        ],
        'description': 'Basic operator access'
    },
    'supervisor': {
        'inherits': ['operator'],
        'permissions': [
            'reports:read',
            'analytics:read',
            'users:read',
            'alerts:manage'
        ],
        'description': 'Supervisor with analytics access'
    },
    'engineer': {
        'inherits': ['supervisor'],
        'permissions': [
            'models:read',
            'models:deploy',
            'system:configure',
            'predictions:write'
        ],
        'description': 'Engineering access for model management'
    },
    'admin': {
        'inherits': ['engineer'],
        'permissions': [
            'users:write',
            'users:delete',
            'system:admin',
            'api_keys:manage'
        ],
        'description': 'Full system administration'
    }
}
```text

### Permission Checking

```python
from src.auth.permissions import check_permission

# Check if user has permission

@check_permission('predictions:write')
def create_prediction(user, sensor_data):

    # Function only executes if user has permission

    return prediction_engine.predict(sensor_data)

# Manual permission check

def manual_check_example(user):
    if user.has_permission('models:deploy'):
        deploy_model()
    else:
        raise PermissionError("Insufficient permissions")
```text

## Multi-Factor Authentication

### Setup MFA

```python

# Enable MFA for user

mfa_setup = requests.post(
    'http://localhost:8000/api/v1/auth/mfa/setup',
    headers=headers,
    json={'method': 'totp'}  # Time-based OTP
)

if mfa_setup.status_code == 200:
    setup_data = mfa_setup.json()
    qr_code_url = setup_data['qr_code']
    backup_codes = setup_data['backup_codes']
    
    print(f"Scan QR code: {qr_code_url}")
    print(f"Backup codes: {backup_codes}")
```text

### MFA Login

```python

# Login with MFA

login_mfa = requests.post(
    'http://localhost:8000/api/v1/auth/login',
    json={
        'username': 'supervisor_001',
        'password': 'secure_password',
        'mfa_code': '123456'  # From authenticator app
    }
)
```text

## Security Headers

### Required Headers

```python

# Security headers for API requests

security_headers = {
    'Authorization': f'Bearer {access_token}',
    'X-Request-ID': 'unique_request_id',
    'X-Client-Version': '1.0.0',
    'User-Agent': 'SteelDefectClient/1.0.0'
}

# Rate limiting headers in response

response_headers = {
    'X-RateLimit-Limit': '1000',
    'X-RateLimit-Remaining': '998',
    'X-RateLimit-Reset': '1640995200'
}
```text

### CORS Configuration

```python

# CORS settings for web applications

cors_config = {
    'allow_origins': [
        'https://dashboard.steel-plant.com',
        'https://mobile.steel-plant.com'
    ],
    'allow_methods': ['GET', 'POST', 'PUT', 'DELETE'],
    'allow_headers': [
        'Authorization',
        'Content-Type',
        'X-API-Key'
    ],
    'expose_headers': [
        'X-RateLimit-Limit',
        'X-RateLimit-Remaining'
    ]
}
```text

## Token Management

### Token Information

```python

# Get token information

token_info = requests.get(
    'http://localhost:8000/api/v1/auth/token/info',
    headers={'Authorization': f'Bearer {access_token}'}
)

token_details = token_info.json()
print(f"Token expires at: {token_details['expires_at']}")
print(f"Issued at: {token_details['issued_at']}")
print(f"Token ID: {token_details['jti']}")
```text

### Token Revocation

```python

# Revoke access token

revoke_response = requests.post(
    'http://localhost:8000/api/v1/auth/revoke',
    json={'token': access_token}
)

# Revoke all user tokens (logout from all devices)

revoke_all = requests.post(
    'http://localhost:8000/api/v1/auth/revoke-all',
    headers={'Authorization': f'Bearer {access_token}'}
)
```text

## Error Handling

### Authentication Errors

```python
def handle_auth_errors(response):
    if response.status_code == 401:
        error_data = response.json()
        
        if error_data['error'] == 'token_expired':

            # Try to refresh token

            return refresh_token_and_retry()
        elif error_data['error'] == 'invalid_token':

            # Redirect to login

            return redirect_to_login()
        elif error_data['error'] == 'mfa_required':

            # Prompt for MFA code

            return prompt_for_mfa()
            
    elif response.status_code == 403:

        # Insufficient permissions

        raise PermissionError("Access denied")
    
    return response
```text

### Retry Logic

```python
import time
from functools import wraps

def with_auth_retry(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    response = func(*args, **kwargs)
                    if response.status_code == 401:

                        # Try to refresh token

                        refresh_access_token()
                        continue
                    return response
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
            
        return wrapper
    return decorator

@with_auth_retry()
def make_api_call(endpoint, data):
    return requests.post(endpoint, json=data, headers=auth_headers)
```text

## API Client Library

### Python Client

```python
from steel_defect_client import SteelDefectClient

# Initialize client with authentication

client = SteelDefectClient(
    base_url='http://localhost:8000',
    username='operator_001',
    password='secure_password'
)

# Client handles authentication automatically

predictions = client.predictions.create(sensor_data)
alerts = client.alerts.list(status='active')
```text

### JavaScript Client

```javascript
// JavaScript SDK with automatic token management
import SteelDefectSDK from 'steel-defect-sdk';

const client = new SteelDefectSDK({
    baseURL: 'http://localhost:8000',
    apiKey: 'your_api_key'
});

// Make authenticated requests
const predictions = await client.predictions.create(sensorData);
const alerts = await client.alerts.list({ status: 'active' });
```text

## Security Best Practices

### Token Security

```python

# Secure token storage

import keyring

# Store token securely

keyring.set_password("steel_defect_system", "access_token", access_token)

# Retrieve token securely

stored_token = keyring.get_password("steel_defect_system", "access_token")
```text

### Network Security

```python

# Use HTTPS for all API calls

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure secure session

session = requests.Session()
session.verify = True  # Verify SSL certificates

# Configure retry strategy

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
```text

This authentication system provides comprehensive security while maintaining ease of use for various integration scenarios.
