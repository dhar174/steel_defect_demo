# Predictive Quality Monitoring System for Continuous Steel Casting (Technical Design)

## 1. Introduction and Context
Continuous (strand) casting solidifies molten steel into semi-finished slabs or billets for downstream rolling. Maintaining high cast quality is challenging because *hundreds* of interacting thermal, mechanical, material, environmental, and equipment condition variables influence defect formation.

KISS Technologies (Holland, MI) focuses on data-driven continuous casting solutions to “take the guesswork out of casting.” This specification defines a proof‑of‑concept (PoC) **Predictive Quality Monitoring System** that performs near real‑time machine learning (ML) inference on streaming caster sensor data to predict quality outcomes (e.g., *defect vs. good* or a continuous quality score) and surface early warnings for operators. The initial PoC uses **synthetic data** while providing a clear migration path to production integration with real plant data infrastructure.

## 2. Scope and Objectives
The system will ingest streaming / batch sensor and contextual data, execute ML models (baseline + sequence model), and expose predictions plus supporting telemetry.

**Key Objectives:**
- **Demonstrate domain understanding** of continuous casting processes and quality determinants.
- **Evaluate multiple modeling paradigms** (feature-based gradient boosted trees, sequence deep learning) and compare performance.
- **Implement a flexible data pipeline** supporting both historical (batch) training and simulated streaming inference.
- **Design for real-time deployment** (low latency, modular, replaceable components).
- **Provide operator-facing outputs** (logs, optional lightweight dashboard) for interpretability and impact.
- **Outline production hardening considerations** (scalability, fault tolerance, monitoring, security, model maintenance).

## 3. Domain Background: Continuous Casting Process & Quality Factors
Molten steel flows from a **ladle** to a **tundish**, then into a water‑cooled **mold** where a solid shell forms around liquid core; the strand is continuously withdrawn, guided by rollers, bent/straightened, secondarily cooled by sprays, and cut to length.

### 3.1 Core Influencing Factors
- **Thermal Conditions:** Superheat above liquidus, mold heat flux, secondary cooling spray patterns, and overall cooling rates influence shell growth and microsegregation. Improper cooling → internal or surface cracks, segregation bands.
- **Mechanical / Operational Parameters:** Casting speed, mold level stability, mold oscillation (frequency, stroke), electromagnetic stirring (if present), withdrawal / straightening forces. Excess fluctuations can entrain slag or induce shell stress.
- **Material Properties:** Steel grade / composition (e.g., carbon, alloying additions), inclusion cleanliness, viscosity-modifying elements; these shift solidification behavior and optimal parameter envelopes.
- **Environmental Factors:** Ambient temperature, humidity, air flows affecting cooling efficiency or water inlet conditions.
- **Equipment Condition:** Oscillator health, roller wear/misalignment, nozzle clogging, pump performance, vibrational anomalies introducing non-uniform heat transfer or mechanical stress.

### 3.2 Quality Defects of Interest
Common defect classes to predict / prevent:
- **Surface cracks** (longitudinal, transverse, corner)
- **Internal cracks** (centerline, subsurface)
- **Corner cracks**
- **Shape deformities** (e.g., rhomboidity—diagonal difference in billet/slab cross-section)
- **Non-metallic inclusions / slag entrapment**

A predictive system aims to *forecast* elevated defect risk in time for corrective actions (adjust speed, cooling, mold level control, oscillation parameters, stirring, etc.).

## 4. Data Sources and Collection Strategy
A robust model integrates multivariate, heterogeneous data streams. For PoC, all signals are synthetically generated; architecture anticipates substitution with real feeds.

### 4.1 Data Source Categories
1. **Process Parameters (Time-Series):** Casting speed, tundish & mold steel temperatures, mold level & fluctuations, mold oscillation frequency/stroke, cooling water flow & temperatures, stopper rod position or slide gate opening, inferred friction metrics.
2. **Material Properties (Per Heat / Sequence):** Steel grade, chemical composition (elemental ranges or categorical ID), inclusion cleanliness index, superheat.
3. **Environmental Data (Slow Time-Series):** Ambient air temperature, humidity, cooling water supply temperature/pressure.
4. **Equipment Condition Signals:** Oscillator vibration / acceleration, roller vibration, hydraulic pressures, motor currents, derived health / maintenance flags.
5. **Production & Quality Outputs (Labels / Targets):** Defect presence (binary), defect class (multi-class future extension), severity metrics (counts, length, shape deviation index), downstream acceptance/rejection, yield metrics.

### 4.2 Data Collection Approach
- **Plant Reality:** PLC / Level 1 sensors aggregate to Level 2 historian or message bus (OPC UA, MQTT, Kafka, PI System). Data consumed by analytics microservices.
- **PoC Simplification:** Python synthetic generator emits per-cast time-series (e.g., 1 Hz) + static context for each cast heat. Outputs:
  - **Training Dataset:** ~1000+ simulated casts with raw time-series + derived features + label.
  - **Real-Time Feed:** Streaming playback (replay test cast at wall time) for inference loop.
- **Structure:** Each cast assigned ID; time-series stored as tall table (timestamp, sensor, value) *or* wide matrix; aggregated features persisted separately for baseline model.
- **Preprocessing:** Normalization (z-score or min-max using train split stats), timestamp alignment, missing-value imputation (forward fill / sentinel), optional smoothing (moving median) on noisy channels.

## 5. System Architecture Overview
**Modular layers ensure replaceability and scalability.**

### 5.1 Components
1. **Data Ingestion Layer:** Connects to synthetic generator (initially CSV iteration / in-memory generator). Production: subscribes to Kafka/MQTT topics or queries historian snapshots.
2. **Data Processing & Feature Engineering:** Rolling window aggregations, statistical feature computation, spike detection, sequence assembly, normalization, missing data strategies.
3. **Model Inference Engine:** Unified interface supporting:
   - *Offline Training Mode* (historical batch)
   - *Online Inference Mode* (streaming / incremental prediction)
   Baselines + deep models swappable.
4. **Alerting & Visualization:** Console logging and optional web dashboard (Plotly Dash / lightweight FastAPI + WebSockets) for real-time sensor + risk curves.
5. **Data Storage & Logging:** Raw synthetic generation artifacts, processed feature tables, trained model artifacts (pickle / Torch script / ONNX), inference logs (JSONL / Parquet).

### 5.2 Architectural Considerations
- **Latency:** Target << 1 s for inference; sequence models operate on buffered windows (e.g., last N seconds) or full-to-date sequence at periodic intervals.
- **Scalability:** Decouple ingestion (stream consumer) from inference (microservice) via message queue; horizontal scale for multi-strand casters.
- **Replaceability:** Swap synthetic ingestion with historian connector without modifying model or visualization modules.
- **Extensibility:** Add anomaly detection microservice feeding derived features to main classifier.

## 6. Modeling Approach: Options Analysis and Recommendations
### 6.1 Problem Formulation
Treat each *cast* (or definable time horizon window) as an instance. Initial focus: **binary classification** (defect vs. no defect). Future: multi-class (specific defect type), regression (severity score), time-to-defect (survival / hazard modeling), or early-warning trajectory (sequence labeling).

### 6.2 Candidate Model Families
1. **Gradient Boosted Decision Trees (XGBoost / LightGBM):** Strong on tabular engineered features, fast training, interpretable feature importances. Limitation: relies on handcrafted summary statistics—may miss temporal micro-patterns.
2. **Classical ML (SVM, k-NN, Logistic Regression):** Baselines; generally inferior to boosted trees here; higher feature engineering burden.
3. **Deep Sequence Models:**
   - **RNN / LSTM / GRU:** Learns temporal dependencies, handles variable-length sequences.
   - **1D CNN:** Captures local temporal motifs (spikes, oscillations); can precede recurrent or transformer layers.
   - **CNN+LSTM Hybrid:** Multi-scale feature extraction + temporal aggregation.
   - **Transformers (Temporal / Time Series Adaptations):** Potential for long-range dependencies; may be overkill for PoC scale but future-proof.
4. **Hybrid / Ensemble:** Combine feature-based GBDT with deep model probability; integrate unsupervised anomaly score (autoencoder reconstruction error, one-class SVM) as meta-feature.

### 6.3 Recommended Strategy
Two-stage implementation:
- **Stage 1 Baseline:** GBDT on engineered per-cast features (statistics, excursions, instability frequencies). Rapid benchmark + interpretability.
- **Stage 2 Sequence Model:** PyTorch LSTM (e.g., stacked 1–2 layers, hidden size ~50–128, dropout) ingesting normalized multivariate sequence → fully connected → sigmoid probability of defect.

Compare metrics (AUC, F1, recall for defect class) to determine value added by temporal modeling. Persist both for demonstration.

### 6.4 Handling Imbalanced Data
Defects likely minority class. Techniques:
- **Class Weights / Weighted Loss** (e.g., BCEWithLogitsLoss with positive class weight).
- **Oversampling / SMOTE (feature model)** or minority-case data augmentation (inject synthetic perturbations in sequences).
- **Evaluation Metrics:** Precision, recall, F1, ROC-AUC, PR-AUC; emphasize recall (capture defects) while controlling false alarm rate.

### 6.5 Feature Engineering (Baseline Model)
Representative feature groups per cast:
- **Statistical:** mean, median, max, min, variance, standard deviation for each sensor.
- **Dynamic / Stability:** Count of spikes above threshold, frequency of mold level excursions, rate of temperature change, oscillation variance.
- **Duration-at-Extremes:** Seconds above high-temp threshold, seconds of unstable mold level, time casting speed outside nominal band.
- **Cross-Interactions:** Products/ratios (speed × superheat, cooling flow balance indices).
- **Anomaly Scores:** From simple autoencoder or isolation forest applied to raw sequence (optional meta-feature).

## 7. Implementation Plan (Step-by-Step)
1. **Environment Setup:** Create Python environment; install `pandas`, `numpy`, `scikit-learn`, `xgboost` or `lightgbm`, `pytorch`, `plotly`/`dash` (optional), `pyyaml` for config. Initialize Git repository; define directory structure (`data/`, `models/`, `src/`, `notebooks/`, `configs/`).
2. **Synthetic Data Generation:** Implement `data_gen.py`:
   - Parameterizable number of casts, duration per cast, sampling rate.
   - Generate realistic baseline signals + stochastic noise + controlled excursions.
   - Deterministic rules mapping excursions (e.g., prolonged mold level deviation, rapid temperature drop + high speed) to defect label.
   - Output raw time-series (Parquet/CSV) and per-cast metadata JSON (grade, composition ID, label).
3. **Exploratory Data Analysis (EDA):** Notebook to visualize exemplar good vs defect sequences; correlation heatmaps; distribution plots; validate labeling logic.
4. **Baseline Feature Engineering & Training:** `features.py` to compute feature table; split dataset (stratified); train GBDT with early stopping; log metrics & feature importances; serialize model + feature scaler.
5. **Deep Sequence Model Development:** `dataset.py` (PyTorch Dataset returns (sequence_tensor, label)); `model_lstm.py`; training script with early stopping on validation AUC; save `state_dict`; optional ONNX export (`torch.onnx.export`) for deployment.
6. **Real-Time Inference Demo:** `simulate_stream.py` replays a cast in (pseudo) real time; incremental buffer updates; periodic (e.g., every 30 simulated seconds) feature recomputation for baseline and sequence forward pass; log probability trajectory; optional live dashboard rendering.
7. **Evaluation & Reporting:** Consolidate metrics, confusion matrices, precision-recall curves; generate brief markdown or HTML summary; identify improvement opportunities (adding new sensors, refining labeling rules, adopting transformer for longer sequences).

## 8. Real-Time Integration and Deployment Considerations
- **Industrial Connectivity:** Replace synthetic generator with OPC UA / MQTT / Kafka ingestion microservice; secure TLS channels; topic partitioning by strand.
- **Scalability & Microservices:** Containerize components: `ingestion`, `feature-service`, `inference-service`, `dashboard`, `model-monitor`; orchestrate with Docker Compose or Kubernetes (future).
- **Model Serving:** FastAPI or TorchServe endpoint; warm-loaded model; batch or single-event inference; ONNX/runtime for portability (CPU inference expected sufficient initially).
- **Data Throughput & Volume:** Use ring buffers for per-strand sequences; downsample low-variance signals; compress archives; partition long-term storage by date + strand.
- **Fault Tolerance:** Graceful degradation (fallback to rules/threshold alarms if ML service unavailable); retry logic; circuit breakers; health probes.
- **Monitoring & Maintenance:** Track data drift (population stability index on features), concept drift (degrading recall), latency metrics, model confidence calibration; schedule periodic retraining pipeline.
- **Security & Access:** Role-based access to dashboards; principle of least privilege; network segmentation; secrets management for credentials.
- **User Interface & Alerting:** Integrate with existing HMI/SCADA; alarm thresholds on predicted risk; trend charts for key sensors + risk index; capability to annotate events (operator feedback loop).

## 9. Conclusion
The specified PoC delivers a production-aligned prototype for predictive quality monitoring in continuous steel casting. It combines interpretable baseline modeling with temporal deep learning to capture dynamic defect precursors, and it establishes a modular pipeline readily extensible to real plant infrastructure. Thoughtful design around data ingestion, processing, inference, visualization, and operational hardening positions the system as both a compelling interview demonstration and a scalable foundation for deployment. Future enhancements can incorporate advanced architectures (transformers, graph-based context modeling), computer vision (surface image inspection), and reinforcement learning for closed-loop parameter optimization.

## 10. References (Indicative)
- Industry domain resources describing continuous casting process parameters and quality impacts.
- Research studies employing ML (GBDT, CNN-LSTM) for inclusion/crack/rhomboidity prediction in continuous casting.
- KISS Technologies public materials on data-driven casting and recommended data categories (process, material, environment, equipment, production).
- General ML deployment best practices (microservices, monitoring, drift detection, model serving frameworks).

*Note: Reference URLs from the original source document can be appended or hyperlinked here as needed.*

