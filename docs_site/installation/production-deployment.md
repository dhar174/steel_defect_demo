# Production Deployment

This guide covers deploying the Steel Defect Prediction System in a production environment.

## Prerequisites

- Linux server with Docker support (Ubuntu 20.04+ recommended)
- Minimum 8GB RAM, 4 CPU cores
- 50GB available disk space
- Network access for API endpoints
- SSL certificates for HTTPS

## Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/dhar174/steel_defect_demo.git
cd steel_defect_demo

# Copy production configuration
cp configs/production.yml.example configs/production.yml

# Edit configuration for your environment
nano configs/production.yml

# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### Option 2: Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/configmap.yml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/ingress.yml
```

## Configuration

### Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/steel_defects
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-here

# Model Configuration
MODEL_PATH=/app/models/production_model.pth
PREDICTION_THRESHOLD=0.7

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
```

### SSL Configuration

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Health Checks

The system provides health check endpoints:

- `/health` - Basic health status
- `/health/detailed` - Detailed component status
- `/metrics` - Prometheus metrics

## Backup Strategy

### Database Backup

```bash
# Daily backup script
#!/bin/bash
pg_dump steel_defects > backup_$(date +%Y%m%d).sql
aws s3 cp backup_$(date +%Y%m%d).sql s3://your-backup-bucket/
```

### Model Backup

```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
aws s3 cp models_backup_$(date +%Y%m%d).tar.gz s3://your-model-bucket/
```

## Security Considerations

- Use TLS 1.2+ for all communications
- Implement API rate limiting
- Enable authentication for admin endpoints
- Regular security updates
- Network segmentation
- Log monitoring and alerting

## Performance Tuning

### Database Optimization

```sql
-- Index optimization for frequent queries
CREATE INDEX idx_sensor_data_timestamp ON sensor_data(timestamp);
CREATE INDEX idx_predictions_cast_id ON predictions(cast_id);
```

### Model Serving Optimization

```python
# Batch prediction configuration
BATCH_SIZE = 32
MAX_BATCH_WAIT_TIME = 100  # milliseconds
WORKER_THREADS = 4
```

## Monitoring and Alerting

Set up monitoring for:

- System resource usage (CPU, memory, disk)
- Application metrics (response time, error rate)
- Model performance (prediction accuracy, drift)
- Database performance
- Alert thresholds

## Troubleshooting

### Common Issues

1. **High memory usage**: Increase container memory limits
2. **Slow predictions**: Check model loading and GPU availability
3. **Database connection issues**: Verify network connectivity and credentials
4. **SSL certificate errors**: Verify certificate validity and configuration

### Log Locations

- Application logs: `/var/log/steel-defect-prediction/app.log`
- Database logs: `/var/log/postgresql/postgresql.log`
- Nginx logs: `/var/log/nginx/access.log`

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.prod.yml
services:
  api:
    image: steel-defect-prediction:latest
    deploy:
      replicas: 3
    environment:
      - LOAD_BALANCER=nginx
```

### Auto-scaling

Configure auto-scaling based on:

- CPU utilization > 70%
- Memory utilization > 80%
- Request queue length > 100