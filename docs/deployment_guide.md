# Deployment Guide

## Steel Defect Prediction System Deployment

This guide provides step-by-step instructions for deploying the Steel Defect Prediction system in various environments.

### Prerequisites

- Python 3.8 or higher
- Docker (for containerized deployment)
- Kubernetes (for orchestrated deployment)
- Git

### Environment Setup

#### 1. Clone Repository

```bash
git clone <repository_url>
cd steel_defect_demo
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Model paths
MODEL_DIR=/app/models
BASELINE_MODEL_PATH=/app/models/baseline/model.pkl
LSTM_MODEL_PATH=/app/models/deep_learning/model.pth

# Data paths
DATA_DIR=/app/data
CONFIG_DIR=/app/configs

# API settings
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Database settings (if applicable)
DATABASE_URL=postgresql://user:password@localhost:5432/steel_defect

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# Logging
LOG_LEVEL=INFO
LOG_DIR=/app/logs
```

### Local Development Deployment

#### 1. Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --verbose
```

#### 2. Train Models

```bash
# Train baseline model
python scripts/train_baseline_model.py --verbose

# Train LSTM model
python scripts/train_lstm_model.py --verbose
```

#### 3. Run Inference Demo

```bash
python scripts/run_inference_demo.py --dashboard --verbose
```

#### 4. Start API Server (Optional)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

#### 1. Build Docker Image

```bash
docker build -t steel-defect-predictor .
```

#### 2. Run Container

```bash
docker run -d \
  --name steel-defect-predictor \
  -p 8000:8000 \
  -p 8050:8050 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/configs:/app/configs \
  --env-file .env \
  steel-defect-predictor
```

#### 3. Using Docker Compose

```bash
docker-compose up -d
```

#### 4. Check Container Status

```bash
docker ps
docker logs steel-defect-predictor
```

### Production Deployment

#### 1. Kubernetes Deployment

Create deployment manifests:

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: steel-defect-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: steel-defect-predictor
  template:
    metadata:
      labels:
        app: steel-defect-predictor
    spec:
      containers:
      - name: steel-defect-predictor
        image: steel-defect-predictor:latest
        ports:
        - containerPort: 8000
        - containerPort: 8050
        env:
        - name: API_HOST
          value: "0.0.0.0"
        - name: API_PORT
          value: "8000"
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: data
          mountPath: /app/data
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: data
        persistentVolumeClaim:
          claimName: data-pvc
```

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: steel-defect-predictor-service
spec:
  selector:
    app: steel-defect-predictor
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: dashboard
    port: 8050
    targetPort: 8050
  type: LoadBalancer
```

Deploy to Kubernetes:
```bash
kubectl apply -f k8s/
```

#### 2. Load Balancer Configuration

TODO: Configure load balancer (nginx, HAProxy, etc.)

#### 3. Database Setup

TODO: Setup production database if required.

#### 4. Monitoring Setup

TODO: Configure monitoring (Prometheus, Grafana, etc.)

### Configuration Management

#### Environment-Specific Configurations

- **Development:** Use local file-based configuration
- **Staging:** Use environment variables with validation
- **Production:** Use secrets management (Kubernetes secrets, HashiCorp Vault, etc.)

#### Model Management

1. **Model Storage:** Store trained models in object storage (S3, GCS, etc.)
2. **Model Versioning:** Use MLflow or similar for model versioning
3. **Model Updates:** Implement blue-green deployment for model updates

### Security Considerations

#### 1. API Security

- Implement authentication (JWT, API keys)
- Use HTTPS in production
- Rate limiting
- Input validation

#### 2. Container Security

- Use non-root user in containers
- Scan images for vulnerabilities
- Minimal base images

#### 3. Network Security

- Network segmentation
- Firewall rules
- VPN for internal access

### Monitoring and Logging

#### 1. Application Monitoring

- Health checks
- Performance metrics
- Model accuracy monitoring
- Data drift detection

#### 2. Infrastructure Monitoring

- Resource utilization
- Container health
- Network monitoring

#### 3. Logging

- Centralized logging (ELK stack, Splunk)
- Log rotation
- Security event logging

### Backup and Recovery

#### 1. Data Backup

- Regular backups of training data
- Model artifacts backup
- Configuration backup

#### 2. Disaster Recovery

- Multi-region deployment
- Backup restoration procedures
- RTO/RPO definitions

### Troubleshooting

#### Common Issues

1. **Model Loading Errors**
   - Check model file paths
   - Verify model compatibility
   - Check disk space

2. **Performance Issues**
   - Monitor resource usage
   - Check model inference latency
   - Optimize batch sizes

3. **Data Issues**
   - Validate input data format
   - Check for missing features
   - Monitor data quality

#### Debug Commands

```bash
# Check container logs
docker logs steel-defect-predictor

# Check resource usage
docker stats

# Check API health
curl http://localhost:8000/health

# Check model status
curl http://localhost:8000/models
```

### Maintenance

#### Regular Tasks

1. Model retraining
2. Performance monitoring
3. Security updates
4. Data cleanup
5. Log rotation

#### Updates and Upgrades

1. Test in staging environment
2. Backup current deployment
3. Rolling updates in production
4. Monitor post-deployment

### Contact and Support

For deployment issues or questions:
- Technical Support: TODO
- Documentation: TODO
- Issue Tracker: TODO