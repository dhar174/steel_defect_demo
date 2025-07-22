# Docker Installation

This guide covers installing and running the Steel Defect Prediction System using Docker containers.

## Prerequisites

- Docker Engine 20.10+ 
- Docker Compose 2.0+
- 8GB RAM minimum
- 20GB free disk space

## Quick Start with Docker

### 1. Pull the Image

```bash
docker pull ghcr.io/dhar174/steel-defect-prediction:latest
```

### 2. Run with Docker

```bash
# Basic run command
docker run -p 8000:8000 \
  -e DATABASE_URL=sqlite:///app/data/steel_defects.db \
  ghcr.io/dhar174/steel-defect-prediction:latest
```

### 3. Access the Application

Open your browser to `http://localhost:8000`

## Docker Compose Setup

### Development Environment

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    image: ghcr.io/dhar174/steel-defect-prediction:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/steel_defects
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=steel_defects
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app

volumes:
  postgres_data:
```

### Start the Stack

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Custom Docker Build

### Building from Source

```bash
# Clone repository
git clone https://github.com/dhar174/steel_defect_demo.git
cd steel_defect_demo

# Build the image
docker build -t steel-defect-prediction .

# Run your custom build
docker run -p 8000:8000 steel-defect-prediction
```

### Dockerfile Overview

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create data directories
RUN mkdir -p /app/data /app/models /app/logs

# Set permissions
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Environment Configuration

### Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/steel_defects
REDIS_URL=redis://localhost:6379/0

# Application
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key

# Model Configuration
MODEL_PATH=/app/models/production_model.pth
BATCH_SIZE=32
PREDICTION_THRESHOLD=0.7

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

### Configuration Files

Mount configuration files:

```bash
docker run -v $(pwd)/configs:/app/configs steel-defect-prediction
```

## Volume Management

### Persistent Data

```bash
# Create named volumes
docker volume create steel_models
docker volume create steel_data

# Use in containers
docker run -v steel_models:/app/models \
           -v steel_data:/app/data \
           steel-defect-prediction
```

### Backup Volumes

```bash
# Backup models
docker run --rm -v steel_models:/data -v $(pwd):/backup \
  alpine tar czf /backup/models_backup.tar.gz -C /data .

# Restore models
docker run --rm -v steel_models:/data -v $(pwd):/backup \
  alpine tar xzf /backup/models_backup.tar.gz -C /data
```

## Networking

### Custom Networks

```bash
# Create custom network
docker network create steel-network

# Run containers on custom network
docker run --network steel-network steel-defect-prediction
```

### Service Discovery

Use Docker Compose service names for internal communication:

```yaml
services:
  app:
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/steel_defects
```

## Security

### Non-root User

```dockerfile
RUN useradd -m -u 1001 appuser
USER appuser
```

### Secrets Management

```bash
# Use Docker secrets
echo "my-secret-key" | docker secret create app_secret_key -

# In compose file
services:
  app:
    secrets:
      - app_secret_key
```

## Multi-stage Builds

```dockerfile
# Build stage
FROM python:3.9 as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "-m", "src.api.main"]
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Use different ports with `-p 8001:8000`
2. **Permission errors**: Check file ownership and permissions
3. **Memory issues**: Increase Docker memory allocation
4. **Network issues**: Verify container networking

### Debugging

```bash
# Enter running container
docker exec -it <container_id> /bin/bash

# View container logs
docker logs <container_id>

# Inspect container
docker inspect <container_id>
```

### Performance Monitoring

```bash
# Container stats
docker stats

# System resource usage
docker system df
docker system prune
```