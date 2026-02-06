# System Architecture: HMM-Based System Call Anomaly Detection

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Data Architecture](#2-data-architecture)
3. [Model Architecture](#3-model-architecture)
4. [Hybrid Model Architecture](#4-hybrid-model-architecture)
5. [Processing Pipeline](#5-processing-pipeline)
6. [Anomaly Detection Mechanism](#6-anomaly-detection-mechanism)
7. [Evaluation Architecture](#7-evaluation-architecture)
8. [Implementation Stack](#8-implementation-stack)
9. [Experiment Design](#9-experiment-design)
10. [Results and Analysis](#10-results-and-analysis)

---

## 1. System Overview

### 1.1 Purpose
- Intrusion detection system based on system call sequence analysis
- Uses ADFA-LD dataset for training and evaluation
- Achieves high detection rate (95.37%) with low false positive rate (2.78%)

### 1.2 Approach
- **Baseline Model**: Single-class HMM (learns normal behavior only)
- **Hybrid Model**: HMM + Random Forest (superior performance)
- **Detection Strategy**: Identifies patterns that deviate from learned normal behavior
- **Target Attacks**: 6 types (Adduser, Hydra_FTP, Hydra_SSH, Java_Meterpreter, Meterpreter, Web_Shell)

### 1.3 Key Features
- Two-stage architecture: Baseline HMM and Hybrid HMM+RF
- Feature engineering: N-grams, statistical features, HMM likelihood
- Optimized pipeline: Data caching, parallel processing
- Comprehensive evaluation: Per-attack metrics, visualizations

---

## 2. Data Architecture

### 2.1 Dataset: ADFA-LD

**Structure:**
```
ADFA-LD/
├── Training_Data_Master/      # Normal data (173 sequences)
│   ├── UTD-0001.txt           # Space-separated system call IDs
│   ├── UTD-0002.txt
│   └── ...
├── Attack_Data_Master/        # Attack data (215 sequences)
│   ├── Adduser_*/             # 30 samples
│   ├── Hydra_FTP_*/           # 28 samples
│   ├── Hydra_SSH_*/           # 40 samples
│   ├── Java_Meterpreter_*/    # 44 samples
│   ├── Meterpreter_*/         # 28 samples
│   └── Web_Shell_*/           # 45 samples
└── Validation_Data_Master/    # Additional validation data
```

### 2.2 Data Characteristics

- **Format**: Text files with space-separated integers
- **Content**: Linux system call IDs
- **Unique System Calls**: 92 types (mapped from original IDs)
- **Sequence Length**: Variable (hundreds to thousands of calls)
- **Window Size**: 500 system calls per sequence

### 2.3 Data Split Strategy

**Normal Data (Training_Data_Master):**
- **Train Set (60%)**: HMM parameter learning (103 sequences)
- **Validation Set (20%)**: Threshold optimization (34 sequences)
- **Test Set (20%)**: Final evaluation (36 sequences)

**Attack Data (Attack_Data_Master):**
- **Train Set (50%)**: Hybrid model training (107 sequences)
- **Test Set (50%)**: Final evaluation (108 sequences)

---

## 3. Model Architecture

### 3.1 Baseline HMM

**Model Type:** Discrete Hidden Markov Model (CategoricalHMM)

**Parameters:**
- **Hidden States**: 10-20 states
  - Represent abstract system behavior states
  - Examples: file I/O, network operations, process management

- **Observations**: 92 system call types
  - Direct mapping from system call IDs to observation symbols

- **Trainable Parameters:**
  - **π (Initial State Distribution)**: [1 × n_states]
  - **A (Transition Matrix)**: [n_states × n_states]
  - **B (Emission Matrix)**: [n_states × 92]

**Training Algorithm:**
- **Baum-Welch Algorithm** (EM-based)
  - Learns π, A, B from normal sequences only
  - Maximum iterations: 100
  - Convergence tolerance: 1e-4

**Inference Algorithm:**
- **Forward Algorithm**
  - Computes log P(O|λ) for given sequence
  - λ = (π, A, B): learned HMM parameters

### 3.2 Baseline HMM Performance

| Metric | Value |
|--------|-------|
| FPR | 2.78% |
| TPR | 22.33% |
| F1-Score | 0.364 |
| Precision | 1.0000 |

**Limitations:**
- Low detection rate (only 22.33%)
- Cannot capture complex feature interactions
- Relies solely on sequential patterns

---

## 4. Hybrid Model Architecture

### 4.1 Overview

The hybrid model combines generative (HMM) and discriminative (Random Forest) approaches for superior performance.

**Architecture:**
```
Input Sequences
    ↓
┌─────────────────┐
│  Stage 1: HMM   │  ← Learn normal behavior
└────────┬────────┘
         ↓
┌─────────────────┐
│ Stage 2: Feature│  ← Extract rich features
│    Extraction   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Stage 3: Random │  ← Final classification
│     Forest      │
└────────┬────────┘
         ↓
   Classification
```

### 4.2 Stage 1: HMM Training

**Purpose:** Learn normal behavior representation

- Train on normal sequences only (103 samples)
- Set threshold at 50th percentile of validation likelihood
- Extract HMM log-likelihood as a feature

### 4.3 Stage 2: Feature Extraction

**Feature Groups:**

**1. HMM-based Feature (1 feature)**
- `hmm_log_likelihood`: Log P(O|λ) from HMM

**2. Statistical Features (18 features)**
- `unique_syscalls`: Number of unique system calls
- `entropy`: Shannon entropy of system call distribution
- `transition_diversity`: Unique state transitions
- `avg_transition_distance`: Average jump distance between calls
- `max_run_length`: Longest consecutive identical calls
- `unique_transitions`: Number of unique call pairs
- And more...

**3. N-gram Features (50 features)**
- **2-grams**: Consecutive call pairs (e.g., `(82, 82)`)
- **3-grams**: Consecutive call triples (e.g., `(82, 82, 82)`)
- **4-grams**: Consecutive call quadruples
- Top 50 most discriminative n-grams selected

**Total Features:** 69 (1 HMM + 18 Statistical + 50 N-gram)

### 4.4 Stage 3: Random Forest Classifier

**Configuration:**
- **n_estimators**: 100 trees
- **max_depth**: 15
- **min_samples_split**: 10
- **min_samples_leaf**: 5
- **class_weight**: 'balanced' (handles imbalanced data)
- **n_jobs**: -1 (parallel training)

**Training Data:**
- Normal: 103 samples (label = 0)
- Attack: 107 samples (label = 1)

**Output:** Binary classification (0 = Normal, 1 = Attack)

### 4.5 Feature Importance

**Top 10 Most Important Features:**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | transition_diversity | 0.1276 |
| 2 | ngram_(82, 82) | 0.1112 |
| 3 | unique_syscalls | 0.0952 |
| 4 | ngram_(82, 82, 82) | 0.0543 |
| 5 | entropy | 0.0333 |
| 6 | ngram_(82, 1) | 0.0327 |
| 7 | ngram_(1, 4) | 0.0286 |
| 8 | hmm_log_likelihood | 0.0242 |
| 9 | ngram_(82, 55, 55) | 0.0224 |
| 10 | avg_transition_distance | 0.0220 |

**Key Insights:**
- Statistical features dominate (transition_diversity, unique_syscalls)
- N-grams capture specific attack patterns
- HMM likelihood contributes but is not dominant

### 4.6 Hybrid Model Performance

| Metric | Value |
|--------|-------|
| **FPR** | **2.78%** ✅ |
| **TPR** | **95.37%** ✅ |
| **F1-Score** | **0.9717** |
| **Precision** | **0.9904** |
| **Accuracy** | **0.9583** |

**Improvement over Baseline:**
- TPR: +73.04 percentage points
- F1-Score: +167% improvement
- Same FPR (2.78%) maintained

---

## 5. Processing Pipeline

### 5.1 Data Preprocessing

**Sequence Windowing:**
- **Fixed window size**: 500 system calls
- **Extraction method**: First 500 calls from each sequence
- **Sequences < 500**: Excluded from dataset

**Rationale:**
- Computational efficiency
- Focus on early behavior patterns (process initialization)
- Consistent input length for model stability

**System Call Mapping:**
```
Original ID (1-340) → Continuous Index (0-91)
Example: syscall 3 (read) → observation 2
```

### 5.2 Data Loading Flow

```
Raw Files (.txt)
    ↓
Parse Sequences (space-delimited)
    ↓
Length Validation (>= 500)
    ↓
Window Extraction (first 500)
    ↓
System Call Mapping (0-91)
    ↓
Train/Val/Test Split
    ↓
Model Input
```

### 5.3 Optimization: Data Caching

**Grid Search Optimization:**
- Load data once per window_size
- Cache preprocessed sequences
- Reuse for multiple experiments
- **Speedup**: 12x faster (36 loads → 3 loads)

---

## 6. Anomaly Detection Mechanism

### 6.1 Baseline HMM Detection

**Threshold Strategy:**
1. Compute log-likelihood for all validation sequences
2. Set threshold at **5th percentile** of validation distribution
3. Sequences below threshold → classified as ATTACK

**Formula:**
```
threshold = percentile(log P(O|λ)_validation, 5%)

if log P(O_test|λ) < threshold:
    classify as ATTACK
else:
    classify as NORMAL
```

**Design Rationale:**
- Low false positive rate priority
- Percentile-based: accounts for normal data variability
- Single-class approach: can detect zero-day attacks

### 6.2 Hybrid Model Detection

**Decision Process:**
1. Extract 69 features from input sequence
2. Feed features to Random Forest classifier
3. Predict probability: P(attack | features)
4. If P(attack) > 0.5 → ATTACK, else NORMAL

**Advantages over Baseline:**
- Captures non-sequential patterns
- Learns discriminative boundaries
- Better generalization to unseen attacks

---

## 7. Evaluation Architecture

### 7.1 Performance Metrics

**Primary Metrics:**

- **FPR (False Positive Rate)**: Key constraint (< 5%)
  ```
  FPR = FP / (FP + TN)
  ```

- **TPR (True Positive Rate, Detection Rate)**: Maximize
  ```
  TPR = TP / (TP + FN)
  ```

- **F1-Score**: Harmonic mean of precision and recall
  ```
  F1 = 2 * (Precision * Recall) / (Precision + Recall)
  ```

**Secondary Metrics:**
- Precision, Recall, Accuracy
- Per-attack detection rates
- Confusion matrix

### 7.2 Confusion Matrix (Hybrid Model)

|  | Predicted Normal | Predicted Attack |
|--|-----------------|-----------------|
| **Actual Normal** | 35 (TN) | 1 (FP) |
| **Actual Attack** | 5 (FN) | 103 (TP) |

### 7.3 Per-Attack Detection Rates

| Attack Type | Samples | Detected | Rate |
|-------------|---------|----------|------|
| Adduser | 15 | 15 | 100.00% |
| Hydra_FTP | 14 | 14 | 100.00% |
| Hydra_SSH | 20 | 20 | 100.00% |
| Java_Meterpreter | 22 | 21 | 95.45% |
| Meterpreter | 14 | 12 | 85.71% |
| Web_Shell | 23 | 21 | 91.30% |

---

## 8. Implementation Stack

### 8.1 Core Libraries

**Machine Learning:**
- `hmmlearn >= 0.3.0`: HMM implementation (Baum-Welch, Forward algorithm)
- `scikit-learn >= 1.0.0`: Random Forest, metrics, preprocessing
- `numpy >= 1.21.0`: Numerical operations
- `pandas >= 1.3.0`: Data manipulation

**Visualization:**
- `matplotlib >= 3.5.0`: Plotting
- `seaborn`: Statistical visualizations

**Optimization:**
- `joblib >= 1.0.0`: Parallel processing, model serialization

### 8.2 Module Structure

```
src/
├── data_loader.py        # Data loading, preprocessing, windowing
├── hmm_model.py          # Baseline HMM implementation
├── hybrid_model.py       # Hybrid HMM + RF classifier
├── feature_extractor.py  # Statistical and N-gram features
├── evaluator.py          # Metrics computation
└── visualizer.py         # Result plotting
```

### 8.3 Key Design Patterns

- **Modular architecture**: Each component is independent
- **Pipeline pattern**: Data flows through processing stages
- **Factory pattern**: Configuration-based model creation
- **Caching**: Preprocessed data reuse for efficiency

---

## 9. Experiment Design

### 9.1 Hyperparameter Grid Search

**Explored Parameters:**
- **window_size**: [300, 500, 700]
- **n_states**: [10, 15, 20]
- **threshold_percentile**: [5%, 10%, 15%, 20%]

**Total Experiments:** 36 (3 × 3 × 4)

**Optimization:**
- Parallel execution: 8 cores
- Data caching: 12x speedup
- Runtime: ~5 minutes

### 9.2 Optimal Configuration

**Baseline HMM:**
- window_size: 500
- n_states: 15
- threshold_percentile: 5%
- Result: FPR 2.78%, TPR 22.33%

**Hybrid Model:**
- window_size: 500
- n_states: 20
- use_ngrams: True
- n_estimators: 100
- Result: FPR 2.78%, TPR 95.37%

### 9.3 Reproducibility

**Ensured by:**
- Fixed random seed: 42
- Explicit data split ratios
- Documented hyperparameters
- Saved model checkpoints

---

## 10. Results and Analysis

### 10.1 Model Comparison

| Model | FPR | TPR | F1 | Precision |
|-------|-----|-----|-----|-----------|
| Baseline HMM | 2.78% | 22.33% | 0.364 | 1.0000 |
| **Hybrid HMM+RF** | **2.78%** | **95.37%** | **0.9717** | **0.9904** |

**Key Findings:**
1. Hybrid model achieves **73%p improvement** in TPR
2. FPR remains under 5% constraint
3. Near-perfect precision (99.04%)
4. F1-score indicates excellent balance

### 10.2 Research Questions Answered

**Q1: Can single-class HMM effectively detect anomalies?**
- ✅ Yes, but limited (22.33% TPR)
- Baseline HMM alone is insufficient

**Q2: How many hidden states are needed?**
- ✅ 15-20 states provide optimal performance
- More states capture finer behavioral nuances

**Q3: Is 500 system calls sufficient?**
- ✅ Yes, sufficient for most attack types
- Early behavior patterns are discriminative

**Q4: What TPR is achievable under FPR ≤ 5%?**
- ✅ 95.37% TPR achieved with hybrid approach
- Exceeds "ideal" target (90%)

### 10.3 Why Hybrid Model Works

**Synergistic Combination:**
1. **HMM**: Captures sequential dependencies
2. **Statistical Features**: Global behavior characteristics
3. **N-grams**: Specific attack signatures
4. **Random Forest**: Non-linear decision boundaries

**Complementary Strengths:**
- HMM: Good at modeling normal sequences
- Features: Expose hidden patterns
- RF: Discriminates between normal and attack features

### 10.4 Failure Analysis

**Missed Attacks (5 False Negatives):**
- Meterpreter: 2 missed (stealth techniques)
- Java_Meterpreter: 1 missed
- Web_Shell: 2 missed (mimics normal web server behavior)

**Improvement Opportunities:**
- Ensemble methods
- Deeper feature engineering
- Attack-specific sub-models

---

## 11. System Constraints and Assumptions

### 11.1 Assumptions

1. **Normal Data Consistency**: Training set represents typical normal behavior
2. **Attack Distinctiveness**: Attack patterns are statistically distinguishable
3. **Stationarity**: System behavior patterns are relatively stable
4. **Label Correctness**: ADFA-LD labels are accurate

### 11.2 Limitations

1. **Fixed Window**: Only first 500 calls analyzed
2. **Discrete HMM**: Ignores timing information
3. **Binary Classification**: No attack type identification
4. **Static Analysis**: No real-time adaptation

### 11.3 Future Enhancements

**Potential Improvements:**
- **Sliding Windows**: Analyze multiple positions
- **Continuous HMM**: Model inter-call timing
- **Deep Learning**: LSTM/Transformer architectures
- **Multi-class**: Identify specific attack types
- **Online Learning**: Adaptive threshold adjustment
- **Ensemble**: Combine multiple models

---

## 12. Architecture Diagrams

### 12.1 Overall System Flow

```
┌──────────────────┐
│  ADFA-LD Dataset │
└────────┬─────────┘
         ↓
┌────────────────────────────┐
│ Data Loader & Preprocessor │
│  - Load sequences          │
│  - Window extraction (500) │
│  - System call mapping     │
└────────┬───────────────────┘
         ↓
┌────────────────────────────┐
│  Train/Val/Test Split      │
│  - Normal: 60/20/20        │
│  - Attack: 50/50           │
└─────┬──────────────────────┘
      ↓
┌─────┴──────────────────────┐
│   Model Training           │
│                            │
│  ┌──────────────────────┐  │
│  │  Baseline HMM        │  │
│  │  - Baum-Welch        │  │
│  │  - Threshold: 5%     │  │
│  └──────────────────────┘  │
│                            │
│  ┌──────────────────────┐  │
│  │  Hybrid Model        │  │
│  │  - HMM               │  │
│  │  - Feature Extract   │  │
│  │  - Random Forest     │  │
│  └──────────────────────┘  │
└─────┬──────────────────────┘
      ↓
┌─────────────────────────────┐
│  Evaluation & Analysis      │
│  - Metrics computation      │
│  - Confusion matrix         │
│  - Per-attack analysis      │
│  - Visualizations           │
└─────┬───────────────────────┘
      ↓
┌─────────────────────────────┐
│  Results & Reports          │
│  - Model checkpoints        │
│  - Performance reports      │
│  - Visualization plots      │
└─────────────────────────────┘
```

### 12.2 Hybrid Model Detail

```
Input: System Call Sequence [500 calls]
           ↓
    ┌──────────────┐
    │  Stage 1:    │
    │  HMM Model   │
    │              │
    │  - Forward   │
    │  - Get λ     │
    └──────┬───────┘
           ↓
    ┌──────────────────────────┐
    │  Stage 2:                │
    │  Feature Extraction      │
    │                          │
    │  ┌──────────────────┐    │
    │  │ HMM Likelihood   │────┤
    │  └──────────────────┘    │
    │  ┌──────────────────┐    │
    │  │ Statistical (18) │────┤──→ [69 features]
    │  └──────────────────┘    │
    │  ┌──────────────────┐    │
    │  │ N-grams (50)     │────┤
    │  └──────────────────┘    │
    └──────┬───────────────────┘
           ↓
    ┌──────────────┐
    │  Stage 3:    │
    │  Random      │
    │  Forest      │
    │  (100 trees) │
    └──────┬───────┘
           ↓
    Binary Classification
    [0: Normal, 1: Attack]
```

---

## Appendix: Terminology

- **HMM**: Hidden Markov Model - probabilistic model with hidden states
- **Forward Algorithm**: Efficient computation of sequence likelihood
- **Baum-Welch**: EM-based algorithm for learning HMM parameters
- **Likelihood**: Probability of observing a sequence given the model
- **FPR**: False Positive Rate - normal sequences misclassified as attacks
- **TPR**: True Positive Rate - attack sequences correctly detected
- **Percentile**: Value below which a percentage of data falls
- **Single-class Learning**: Learning from one class (normal) only
- **N-gram**: Sequence of N consecutive system calls
- **Feature Engineering**: Creating informative features from raw data
- **Ensemble**: Combining multiple models for better performance
- **Random Forest**: Ensemble of decision trees
- **Bagging**: Bootstrap aggregating for ensemble learning

---

## References

1. ADFA-LD Dataset: Australian Defence Force Academy Linux Dataset
2. hmmlearn: Hidden Markov Models in Python
3. scikit-learn: Machine Learning in Python
4. Forrest et al. (1996): "A Sense of Self for Unix Processes"
5. Warrender et al. (1999): "Detecting Intrusions Using System Calls"

---

**Document Version:** 2.0
**Last Updated:** 2026-02-06
**Status:** Production
