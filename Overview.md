# ML System Design Preparation Guide

## Overview

ML System Design interviews test your ability to design end-to-end machine learning systems that work in production. This is different from coding interviews or pure ML theory questions.

---

## 1. Core Framework: The 6-Step Approach

### Step 1: Clarify Requirements (5-10 min)

**Ask clarifying questions:**

- **Business objective:** What problem are we solving? What's the success metric?
- **Scale:** How many users? QPS (queries per second)? Data volume?
- **Latency requirements:** Real-time (< 100ms)? Near real-time (< 1s)? Batch?
- **Constraints:** Budget? Existing infrastructure? Privacy requirements?
- **Personalization:** User-specific or generic model?

**Example questions for "Design a recommendation system for YouTube":**

- What's the primary goal? (Watch time, CTR, user satisfaction?)
- How many videos? How many users? How many requests per second?
- Real-time recommendations or can they be pre-computed?
- Do we need to handle cold start (new users/videos)?

### Step 2: Define ML Problem (5 min)

**Convert business problem to ML problem:**

- **Problem type:** Classification, regression, ranking, generation?
- **Input/Output:** What are the features? What are we predicting?
- **Success metrics:**
    - ML metrics: Precision, Recall, F1, AUC, NDCG, etc.
    - Business metrics: Revenue, engagement, retention

**Example for YouTube recommendations:**

- **Type:** Ranking problem
- **Input:** User history, video metadata, context
- **Output:** Ranked list of videos
- **Metrics:** Click-through rate, watch time, user engagement

### Step 3: Data Pipeline (10 min)

**Design how data flows:**

- **Data sources:** Where does data come from?
- **Data collection:** Logging, instrumentation
- **Data storage:** Data lakes, warehouses, feature stores
- **Feature engineering:** What features to extract?
- **Data quality:** Handling missing values, outliers, data validation
- **Train/test split:** How to split (temporal, random, stratified)?

**Key considerations:**

- Historical data for training
- Real-time features for inference
- Feature freshness and staleness
- Sampling strategies for large datasets

### Step 4: Model Development (10-15 min)

**Choose and design models:**

#### Model Selection

- **Start simple:** Baseline models (logistic regression, decision trees)
- **Then complex:** Deep learning if justified
- **Consider alternatives:**
    - Collaborative filtering (matrix factorization, neural collaborative filtering)
    - Content-based filtering
    - Hybrid approaches
    - Two-tower models, transformers

#### Model Architecture

- **Input layer:** How to encode features?
- **Hidden layers:** Architecture choices
- **Output layer:** Final predictions
- **Loss function:** What are we optimizing?

#### Training Strategy

- **Offline training:** Batch training on historical data
- **Online learning:** Continuous updates with new data
- **Transfer learning:** Pre-trained models
- **Multi-task learning:** Learning multiple objectives

**Example for recommendations:**

```
Candidate Generation (Retrieval):
- Two-tower model or ANN for fast retrieval from millions of items
- Reduces to ~hundreds of candidates

Ranking:
- Deep neural network with user/item/context features
- Predicts engagement probability
- Ranks top ~10-50 items
```

### Step 5: System Architecture (15 min)

**Design the complete system:**

#### Components:

1. **Data pipeline**
    
    - Batch processing (Spark, Airflow)
    - Stream processing (Kafka, Flink)
    - Feature store (Feast, Tecton)
2. **Training pipeline**
    
    - Experiment tracking (MLflow, Weights & Biases)
    - Model versioning
    - Hyperparameter tuning
    - Distributed training if needed
3. **Serving infrastructure**
    
    - Model serving (TensorFlow Serving, TorchServe)
    - API gateway
    - Caching layer (Redis)
    - Load balancing
4. **Monitoring & feedback loop**
    
    - Model performance monitoring
    - Data drift detection
    - A/B testing framework
    - Retraining triggers

#### Architecture Diagram Example:

```
User Request → API Gateway → Feature Service → Model Service → Response
                                    ↓                ↓
                              Feature Store    Model Registry
                                    ↑                ↑
                              ETL Pipeline   Training Pipeline
                                    ↑                ↑
                                Raw Data      Training Data
```

### Step 6: Evaluation & Iteration (5-10 min)

**How to validate and improve:**

#### Offline Evaluation

- **Metrics:** Precision@K, Recall@K, NDCG, AUC-ROC
- **Cross-validation:** Time-based splits for temporal data
- **Backtesting:** Test on historical data

#### Online Evaluation

- **A/B testing:** Compare model variants
- **Interleaving:** Mix results from different models
- **Multi-armed bandits:** Adaptive experimentation

#### Monitoring

- **Model metrics:** Prediction distribution, calibration
- **System metrics:** Latency, throughput, error rates
- **Business metrics:** Revenue, engagement, retention
- **Data quality:** Feature drift, data freshness

#### Iteration

- **Retraining frequency:** Daily? Weekly? Triggered?
- **Model updates:** Gradual rollout, canary deployment
- **Failure handling:** Fallback to simpler model or cache

---

## 2. Common ML System Design Questions

### Recommendation Systems

- YouTube video recommendations
- Netflix movie recommendations
- Amazon product recommendations
- Spotify music recommendations

**Key points:**

- Candidate generation + ranking (two-stage)
- Cold start problem (new users/items)
- Exploration vs exploitation
- Diversity vs relevance

### Search & Ranking

- Google search ranking
- LinkedIn job search
- Airbnb search ranking

**Key points:**

- Query understanding
- Document retrieval
- Learning to rank
- Personalization vs relevance

### Computer Vision

- Self-driving car perception
- Face recognition system
- Content moderation (detecting inappropriate images)

**Key points:**

- Real-time constraints
- Model size vs accuracy tradeoff
- Edge deployment
- Data annotation pipeline

### NLP Systems

- Machine translation
- Chatbot/conversational AI
- Sentiment analysis
- Spam detection

**Key points:**

- Pre-trained models (BERT, GPT)
- Fine-tuning strategy
- Handling multiple languages
- Context window limitations

### Ads & Feed Ranking

- Facebook news feed ranking
- Twitter timeline
- Ad CTR prediction

**Key points:**

- Multi-objective optimization (relevance, diversity, monetization)
- Position bias
- Auction mechanisms
- Real-time bidding

### Forecasting

- Demand forecasting
- Stock price prediction
- Weather prediction

**Key points:**

- Time series modeling
- Seasonality and trends
- Handling missing data
- Forecast horizons

---

## 3. Key ML Concepts to Master

### ML Fundamentals

- Supervised vs unsupervised learning
- Classification vs regression
- Overfitting vs underfitting
- Bias-variance tradeoff
- Cross-validation strategies
- Feature engineering
- Regularization (L1, L2, dropout)
- Ensemble methods

### Deep Learning

- Neural network architectures (CNN, RNN, Transformer)
- Activation functions
- Optimization algorithms (SGD, Adam)
- Batch normalization, layer normalization
- Attention mechanisms
- Transfer learning
- Fine-tuning strategies

### Specialized Models

- **Recommender systems:** Collaborative filtering, matrix factorization, neural CF
- **Ranking:** Learning to rank (pointwise, pairwise, listwise)
- **NLP:** BERT, GPT, T5, word embeddings
- **Computer Vision:** ResNet, YOLO, Vision Transformers
- **Time Series:** ARIMA, Prophet, LSTMs

### ML at Scale

- Distributed training (data parallelism, model parallelism)
- Batch vs online learning
- Feature stores
- Model compression (quantization, pruning, distillation)
- Edge deployment
- Serving optimization

---

## 4. System Design Concepts

### Scalability

- **Horizontal vs vertical scaling**
- **Sharding and partitioning**
- **Caching strategies** (Redis, Memcached)
- **Load balancing**
- **Database choices** (SQL vs NoSQL)

### Reliability

- **Failure modes and handling**
- **Redundancy and replication**
- **Circuit breakers**
- **Graceful degradation**
- **Fallback mechanisms**

### Latency & Throughput

- **P50, P90, P99 latencies**
- **Batching requests**
- **Async processing**
- **CDNs for static content**
- **Model serving optimization**

### Monitoring & Observability

- **Logging (structured logs)**
- **Metrics (Prometheus, Grafana)**
- **Tracing (Jaeger, Zipkin)**
- **Alerting**
- **Dashboards**

---

## 5. Common Tradeoffs to Discuss

### Model Complexity vs Latency

- Simple models: Fast, interpretable, less accurate
- Complex models: Slow, black-box, more accurate
- **Solution:** Two-stage (fast retrieval + complex ranking)

### Accuracy vs Fairness

- Optimizing for accuracy might introduce bias
- **Solution:** Fairness constraints, debiasing techniques

### Personalization vs Privacy

- More data = better personalization, worse privacy
- **Solution:** Federated learning, differential privacy

### Exploration vs Exploitation

- Explore: Try new items (learn more, might show bad items)
- Exploit: Show known good items (safer, miss opportunities)
- **Solution:** Multi-armed bandits, epsilon-greedy

### Freshness vs Stability

- Frequent updates: Fresh but unstable
- Infrequent updates: Stable but stale
- **Solution:** Gradual rollout, A/B testing

### Batch vs Real-time

- Batch: Efficient, delayed
- Real-time: Immediate, expensive
- **Solution:** Lambda architecture (batch + stream)

---

## 6. How to Practice

### Study Resources

1. **Books:**
    
    - "Designing Machine Learning Systems" by Chip Huyen
    - "Machine Learning System Design Interview" by Ali Aminian & Alex Xu
    - "Designing Data-Intensive Applications" by Martin Kleppmann
2. **Online courses:**
    
    - System Design Interview courses (Educative, Exponent)
    - ML production courses (Made With ML, Full Stack Deep Learning)
3. **Practice platforms:**
    
    - Leetcode ML questions
    - interviewing.io
    - Pramp (peer practice)

### Mock Interview Practice

1. **Pick a problem** (e.g., "Design Instagram feed ranking")
2. **Set a timer** (45 minutes)
3. **Go through the 6 steps** systematically
4. **Draw diagrams** (architecture, data flow)
5. **Explain tradeoffs** clearly
6. **Review:** What did you miss? What could be better?

### Build a Portfolio

- Implement small ML systems end-to-end
- Deploy a model to production (even hobby project)
- Write blog posts explaining your design decisions
- Contribute to open-source ML projects

---

## 7. Interview Tips

### Communication

- **Think out loud:** Explain your reasoning
- **Ask clarifying questions:** Don't make assumptions
- **Structure your answer:** Use the framework
- **Be honest:** Say "I don't know" if you don't, then reason through it

### Time Management

- Clarification: 5-10 min
- ML problem formulation: 5 min
- High-level design: 10 min
- Deep dive: 15-20 min
- Discussion/questions: 5-10 min

### Common Mistakes to Avoid

- Jumping to solutions without clarifying requirements
- Focusing only on ML, ignoring system design
- Not discussing tradeoffs
- Over-engineering for small scale
- Under-engineering for large scale
- Ignoring monitoring and failure modes
- Not considering data quality issues

### What Interviewers Look For

1. **Problem formulation:** Can you translate business to ML problem?
2. **System thinking:** Do you consider the whole system, not just the model?
3. **Tradeoff awareness:** Do you understand different approaches and their pros/cons?
4. **Scalability:** Can your design handle growth?
5. **Practicality:** Is your solution implementable and maintainable?
6. **Communication:** Can you explain complex ideas clearly?

---

## 8. Sample Problem Walkthrough

### Problem: Design a YouTube Video Recommendation System

#### Step 1: Requirements (5 min)

**Q:** Clarify with interviewer

- Users: 2 billion monthly active users
- Videos: 500M+ videos, 500 hours uploaded per minute
- Goal: Maximize watch time
- Latency: < 500ms for recommendations
- Personalized recommendations

#### Step 2: ML Problem (3 min)

- **Type:** Ranking problem (retrieval + ranking)
- **Input:** User history, video metadata, context (time, device)
- **Output:** Ranked list of ~10 videos
- **Metrics:**
    - Offline: Precision@10, NDCG
    - Online: CTR, watch time per user

#### Step 3: Data (8 min)

**Data sources:**

- User interactions (views, likes, comments, watch time)
- Video metadata (title, description, category, upload time)
- User profile (demographics, subscriptions)
- Context (time of day, device, location)

**Features:**

- User features: watch history, engagement rate, demographics
- Video features: popularity, recency, category, quality
- Context features: time, device, location
- Interaction features: user-video affinity

#### Step 4: Model (12 min)

**Two-stage approach:**

**Stage 1: Candidate Generation (Retrieval)**

- Goal: Reduce 500M videos to ~1000 candidates
- Approach: Two-tower neural network
    - User tower: Embeds user features
    - Video tower: Embeds video features
    - Similarity: Dot product
- Fast retrieval: ANN (Approximate Nearest Neighbors)

**Stage 2: Ranking**

- Goal: Rank 1000 candidates to top 10
- Approach: Deep neural network
    - Input: User, video, context features
    - Architecture: Deep & Wide or Transformer
    - Output: Predicted watch time
- Loss: Weighted logistic regression (watch time as weight)

#### Step 5: Architecture (15 min)

```
User Request
    ↓
API Gateway
    ↓
Feature Service ← Feature Store (Redis)
    ↓
Candidate Generation (ANN) → 1000 candidates
    ↓
Ranking Model → Top 10 videos
    ↓
Response

Offline:
User/Video Data → ETL (Spark) → Feature Store
                                    ↓
                          Training Pipeline
                                    ↓
                            Model Registry
```

**Key components:**

- **Feature Store:** Pre-computed features (user embeddings, video stats)
- **Model Serving:** TensorFlow Serving with batching
- **Caching:** Cache popular recommendations
- **A/B Testing:** Gradual rollout of new models

#### Step 6: Evaluation (7 min)

**Offline:**

- Temporal train/test split
- Metrics: Precision@10, Recall@10, NDCG

**Online:**

- A/B test: New model vs baseline
- Metrics: CTR, watch time, retention
- Monitor: Prediction distribution, latency

**Monitoring:**

- Model drift: Distribution of predictions
- Data drift: Feature distributions
- Performance: P99 latency, QPS
- Business: Revenue, engagement

**Iteration:**

- Retrain: Daily (new videos, user behavior changes)
- Update: Gradual rollout (1% → 10% → 100%)
- Fallback: Previous model if metrics degrade

---

## 9. Quick Reference Checklist

**Before the interview:**

- [ ] Review ML fundamentals
- [x] Practice system design framework
- [ ] Study common architectures
- [ ] Review scaling concepts
- [ ] Practice drawing diagrams

**During the interview:**

- [ ] Clarify requirements thoroughly
- [ ] Define success metrics clearly
- [ ] Consider scale from the start
- [ ] Discuss tradeoffs explicitly
- [ ] Draw architecture diagrams
- [ ] Think about failure modes
- [ ] Plan for monitoring
- [ ] Be ready to deep dive on any component

**Key aspects to always cover:**

- [ ] Data pipeline
- [ ] Model selection & architecture
- [ ] Training strategy
- [ ] Serving infrastructure
- [ ] Monitoring & evaluation
- [ ] Scalability
- [ ] Failure handling

---

Good luck with your ML system design preparation!