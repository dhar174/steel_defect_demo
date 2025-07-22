# System Architecture Overview

This document provides a comprehensive overview of the Steel Defect Prediction System architecture, including components, data flow, and integration patterns.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A[Sensor Data]
        B[Historical Data]
        C[Configuration]
    end
    
    subgraph "Data Pipeline"
        D[Data Ingestion]
        E[Feature Engineering]
        F[Data Validation]
        G[Data Storage]
    end
    
    subgraph "ML Pipeline"
        H[Model Training]
        I[Model Validation]
        J[Model Deployment]
        K[Inference Engine]
    end
    
    subgraph "Application Layer"
        L[Prediction API]
        M[Dashboard Backend]
        N[Alert System]
        O[Monitoring]
    end
    
    subgraph "User Interface"
        P[Web Dashboard]
        Q[API Clients]
        R[Alert Notifications]
    end
    
    A --> D
    B --> D
    C --> D
    
    D --> E
    E --> F
    F --> G
    
    G --> H
    H --> I
    I --> J
    J --> K
    
    K --> L
    K --> M
    K --> N
    
    L --> Q
    M --> P
    N --> R
    O --> P
    
    style A fill:#e1f5fe
    style P fill:#e8f5e8
    style K fill:#fff3e0
    style N fill:#ffebee
```

## Component Architecture

### 1. Data Layer

#### Data Sources
- **Sensor Systems**: Real-time sensor data streams
- **Historical Database**: Historical casting data
- **Configuration Store**: System and model configurations
- **External Systems**: ERP, MES integration

#### Data Storage
```
data/
├── raw/                 # Raw sensor data
├── processed/          # Cleaned and validated data
├── features/           # Engineered features
├── models/             # Trained model artifacts
└── results/            # Prediction results
```

### 2. Processing Layer

#### Data Pipeline Components

```mermaid
graph LR
    A[Raw Data] --> B[Validation]
    B --> C[Cleaning]
    C --> D[Feature Engineering]
    D --> E[Normalization]
    E --> F[Storage]
    
    subgraph "Quality Checks"
        G[Range Validation]
        H[Anomaly Detection]
        I[Completeness Check]
    end
    
    B --> G
    B --> H
    B --> I
```

#### Key Components
- **Data Ingestion**: `src/connectors/data_connectors.py`
- **Feature Engineering**: `src/features/feature_engineering.py`
- **Data Validation**: `src/data/data_validation.py`
- **Storage Management**: `src/data/data_storage.py`

### 3. Machine Learning Layer

#### Model Architecture

```mermaid
graph TB
    subgraph "Training Pipeline"
        A[Training Data]
        B[Feature Selection]
        C[Model Training]
        D[Validation]
        E[Model Registry]
    end
    
    subgraph "Models"
        F[XGBoost Baseline]
        G[LSTM Deep Learning]
        H[Ensemble Model]
    end
    
    subgraph "Inference Pipeline"
        I[Real-time Features]
        J[Model Ensemble]
        K[Prediction Output]
        L[Confidence Scoring]
    end
    
    A --> B
    B --> C
    C --> F
    C --> G
    C --> H
    
    F --> E
    G --> E
    H --> E
    
    I --> J
    F --> J
    G --> J
    H --> J
    
    J --> K
    J --> L
```

#### Model Components
- **Baseline Model**: XGBoost classifier with engineered features
- **Deep Learning**: LSTM for sequence modeling
- **Ensemble**: Weighted combination of multiple models
- **Feature Engineering**: Automated feature extraction and selection

### 4. Application Layer

#### Service Architecture

```python
# Core application structure
src/
├── models/              # ML model implementations
│   ├── baseline_model.py
│   ├── lstm_model.py
│   └── model_trainer.py
├── inference/           # Prediction engine
│   ├── prediction_engine.py
│   └── inference_pipeline.py
├── visualization/       # Dashboard components
│   ├── dashboard.py
│   └── components/
├── monitoring/          # System monitoring
│   ├── alert_system.py
│   └── health_checks.py
└── utils/              # Shared utilities
    ├── config.py
    └── logging.py
```

#### API Design

```mermaid
graph LR
    subgraph "API Layer"
        A[FastAPI App]
        B[Authentication]
        C[Rate Limiting]
        D[Request Validation]
    end
    
    subgraph "Business Logic"
        E[Prediction Service]
        F[Data Service]
        G[Alert Service]
        H[Monitoring Service]
    end
    
    subgraph "Data Access"
        I[Model Repository]
        J[Data Repository]
        K[Config Repository]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    D --> F
    D --> G
    D --> H
    
    E --> I
    F --> J
    G --> K
    H --> J
```

### 5. User Interface Layer

#### Dashboard Architecture

The dashboard is built using Dash (Plotly) with a component-based architecture:

```python
# Dashboard component structure
src/visualization/
├── dashboard.py         # Main dashboard app
├── components/          # Reusable UI components
│   ├── __init__.py
│   ├── prediction_display.py
│   ├── model_comparison.py
│   ├── historical_analysis.py
│   ├── alert_management.py
│   └── sensor_monitoring.py
├── layouts/            # Page layouts
├── callbacks/          # Interactive callbacks
└── utils/             # UI utilities
```

## Data Flow Architecture

### Real-time Prediction Flow

```mermaid
sequenceDiagram
    participant S as Sensors
    participant DC as Data Connector
    participant FE as Feature Engine
    participant IE as Inference Engine
    participant DB as Dashboard
    participant AS as Alert System
    
    S->>DC: Raw sensor data
    DC->>FE: Validated data
    FE->>IE: Engineered features
    IE->>IE: Model inference
    IE->>DB: Prediction results
    IE->>AS: Alert evaluation
    AS->>AS: Threshold check
    DB->>DB: UI update
    
    Note over IE: Multiple model<br/>ensemble
    Note over AS: Configurable<br/>thresholds
```

### Batch Processing Flow

```mermaid
sequenceDiagram
    participant HD as Historical Data
    participant DP as Data Pipeline
    participant MT as Model Trainer
    participant MR as Model Registry
    participant IE as Inference Engine
    
    HD->>DP: Historical dataset
    DP->>DP: Feature engineering
    DP->>MT: Training data
    MT->>MT: Model training
    MT->>MR: Model artifacts
    MR->>IE: Updated models
    
    Note over MT: Cross-validation<br/>and hyperparameter<br/>optimization
```

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|------------|---------|
| **ML Framework** | PyTorch, scikit-learn, XGBoost | Model development and training |
| **Data Processing** | pandas, NumPy, PyArrow | Data manipulation and analysis |
| **Web Framework** | Dash, Plotly | Interactive dashboard |
| **API Framework** | FastAPI | RESTful API services |
| **Database** | SQLite/PostgreSQL | Data persistence |
| **Caching** | Redis | High-performance caching |

### Development Tools

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Testing** | pytest, unittest |
| **Linting** | flake8, black, mypy |
| **Documentation** | MkDocs, Sphinx |
| **Containerization** | Docker, docker-compose |
| **CI/CD** | GitHub Actions |

## Deployment Architecture

### Development Environment

```yaml
# Local development stack
services:
  app:
    build: .
    ports:
      - "8050:8050"
    volumes:
      - .:/app
    environment:
      - ENV=development
      
  database:
    image: postgres:13
    environment:
      - POSTGRES_DB=steel_defect
      
  redis:
    image: redis:6-alpine
```

### Production Environment

```mermaid
graph TB
    subgraph "Load Balancer"
        A[HAProxy/nginx]
    end
    
    subgraph "Application Tier"
        B[App Instance 1]
        C[App Instance 2]
        D[App Instance N]
    end
    
    subgraph "Data Tier"
        E[PostgreSQL Primary]
        F[PostgreSQL Replica]
        G[Redis Cluster]
    end
    
    subgraph "Monitoring"
        H[Prometheus]
        I[Grafana]
        J[ELK Stack]
    end
    
    A --> B
    A --> C
    A --> D
    
    B --> E
    C --> E
    D --> E
    
    E --> F
    
    B --> G
    C --> G
    D --> G
    
    B --> H
    C --> H
    D --> H
    
    H --> I
    H --> J
```

## Security Architecture

### Authentication and Authorization

```mermaid
graph LR
    subgraph "Client"
        A[Web Browser]
        B[API Client]
    end
    
    subgraph "Auth Layer"
        C[Authentication Service]
        D[Authorization Service]
        E[Token Management]
    end
    
    subgraph "Application"
        F[Dashboard]
        G[API Endpoints]
        H[Admin Interface]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    E --> H
```

### Security Measures

- **Authentication**: Token-based authentication
- **Authorization**: Role-based access control (RBAC)
- **Data Encryption**: TLS/SSL for data in transit
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: API rate limiting and throttling
- **Audit Logging**: Security event logging

## Scalability Considerations

### Horizontal Scaling

- **Stateless Design**: Application instances are stateless
- **Load Balancing**: Multiple app instances behind load balancer
- **Database Scaling**: Read replicas and connection pooling
- **Caching Strategy**: Distributed caching with Redis cluster

### Performance Optimization

- **Model Optimization**: Model quantization and optimization
- **Data Pipeline**: Parallel processing and streaming
- **Frontend**: Lazy loading and caching
- **API**: Async processing and connection pooling

## Integration Patterns

### External System Integration

```mermaid
graph LR
    subgraph "Steel Defect System"
        A[API Gateway]
        B[Data Connectors]
        C[Event Bus]
    end
    
    subgraph "External Systems"
        D[MES System]
        E[ERP System]
        F[SCADA]
        G[Quality Management]
    end
    
    D --> B
    E --> B
    F --> B
    G --> A
    
    B --> C
    A --> C
```

### Integration Methods

- **REST APIs**: Standard HTTP/JSON APIs
- **WebSockets**: Real-time data streaming
- **Message Queues**: Asynchronous processing
- **File Transfer**: Batch data exchange
- **Database Integration**: Direct database connections

---

Next: [Contributing Guide →](../development/contributing.md)