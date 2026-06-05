# Music Trend Prediction — AWS SageMaker

> An end-to-end ML pipeline that predicts whether a music track will trend on streaming platforms, deployed on AWS SageMaker. Built during a data analyst internship at Baboon Forest Entertainment.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Pipeline Architecture](#pipeline-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Features Used](#features-used)
- [Model Performance](#model-performance)
- [Running the Pipeline](#running-the-pipeline)
- [What I Learned](#what-i-learned)

---

## Overview

Music streaming platforms surface thousands of new tracks every week. Labels and independent artists need to know early: _will this track gain traction?_

This project builds a binary classification model that predicts whether a newly released track will enter the top-trending tier within 30 days — based on its audio characteristics, release metadata, and early engagement signals.

The model was trained on historical streaming data, deployed on AWS SageMaker for live scoring, and its insights directly informed content strategy decisions at Baboon Forest Entertainment.

---

## Problem Statement

The challenge: use the information available at release time (audio features, artist profile, metadata) to predict future trending performance — before organic discovery kicks in.

This matters because:

- Promotion budgets are finite — teams need to prioritise which tracks to push
- Playlist pitching has a narrow window (the first 48–72 hours after release)
- Early data signals (saves, playlist adds, skip rates) are powerful predictors if captured and modelled correctly

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Data Pipeline                             │
│                                                              │
│  Raw data      →  ETL (clean,   →  Feature      →  S3       │
│  (streaming        validate,       engineering     (feature  │
│  platform CSV)     transform)      & encoding      store)    │
└──────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────┐
│                  Training Pipeline (SageMaker)               │
│                                                              │
│  S3 data  →  SageMaker   →  Model         →  S3             │
│              Training        evaluation      (model          │
│              Job             & validation    artifact)       │
└──────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────┐
│                   Inference Pipeline                         │
│                                                              │
│  New track  →  SageMaker   →  Trending       →  Content     │
│  metadata      Endpoint        probability       strategy    │
│                                & decision        decision    │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer               | Technology                                      |
| ------------------- | ----------------------------------------------- |
| ML Training         | AWS SageMaker (managed training jobs)           |
| Model               | Scikit-learn (Random Forest + XGBoost ensemble) |
| Data Storage        | AWS S3                                          |
| ETL                 | Python, Pandas                                  |
| Feature Engineering | Pandas, NumPy                                   |
| Visualisation       | Matplotlib, Seaborn                             |
| Notebook            | Jupyter                                         |

---

## Project Structure

```
music-trend-prediction/
│
├── pipeline/
│   ├── etl.py              # Data cleaning and validation
│   ├── features.py         # Feature engineering
│   └── train.py            # SageMaker training script
│
├── notebooks/
│   └── music_trend_prediction.ipynb   # Full walkthrough (start here)
│
├── data/
│   └── sample_tracks.csv   # 100-row sample for local testing
│
├── requirements.txt
└── README.md
```

---

## Features Used

### Audio features (from streaming platform API)

| Feature        | Description                              |
| -------------- | ---------------------------------------- |
| `tempo_bpm`    | Beats per minute                         |
| `energy`       | Intensity and activity (0.0–1.0)         |
| `danceability` | How suitable for dancing (0.0–1.0)       |
| `valence`      | Musical positiveness/happiness (0.0–1.0) |
| `acousticness` | Acoustic vs. electronic (0.0–1.0)        |
| `duration_sec` | Track length in seconds                  |

### Release metadata

| Feature               | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `release_day_of_week` | Friday releases outperform mid-week                  |
| `is_friday_release`   | Binary flag (Friday = industry standard release day) |
| `release_month`       | Seasonal effects (December, summer)                  |

### Artist profile signals

| Feature                       | Description                        |
| ----------------------------- | ---------------------------------- |
| `artist_followers`            | Platform follower count at release |
| `artist_prev_trending_tracks` | Artist's historical trending count |
| `days_since_last_release`     | Release cadence                    |

### Early engagement (first 48 hours)

| Feature                 | Description                              |
| ----------------------- | ---------------------------------------- |
| `streams_48h`           | Total streams in first 48 hours          |
| `save_rate_48h`         | Saves / streams (listener intent signal) |
| `playlist_add_rate_48h` | Editorial and user playlist adds         |

---

## Model Performance

| Metric                     | Value |
| -------------------------- | ----- |
| Accuracy                   | ~87%  |
| Precision (trending class) | 81%   |
| Recall (trending class)    | 79%   |
| ROC-AUC                    | 0.91  |

**Most predictive features:** `save_rate_48h` > `streams_48h` > `artist_prev_trending_tracks` > `danceability` > `playlist_add_rate_48h`

---

## Running the Pipeline

### Local (without SageMaker)

```bash
# 1. Clone and install
git clone https://github.com/okeiylod94/Music-Trend-Prediction.git
cd music-trend-prediction
pip install -r requirements.txt

# 2. Run the notebook
jupyter notebook notebooks/music_trend_prediction.ipynb
```

### With AWS SageMaker

```python
import sagemaker
from sagemaker.sklearn import SKLearn

# Point to your training data in S3
estimator = SKLearn(
    entry_point='pipeline/train.py',
    role=sagemaker.get_execution_role(),
    instance_type='ml.m5.large',
    framework_version='1.0-1'
)

estimator.fit({'train': 's3://your-bucket/music-features/train.csv'})

# Deploy for real-time predictions
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)
```

---

## What I Learned

- **Early engagement signals dominate** — the 48-hour save rate was a stronger predictor than any audio feature. The model is essentially learning "did early listeners love it enough to save it?"
- **Class imbalance is a real problem** — only ~8% of tracks trend. Without SMOTE oversampling and `class_weight='balanced'`, the model just predicted "won't trend" for everything and hit 92% accuracy while being useless
- **SageMaker's managed training is genuinely convenient** — being able to kick off a training job with a single API call, have it run on a proper instance, and get the artifact back in S3 without managing infrastructure is valuable
- **Feature leakage is easy to introduce by accident** — early versions of this model accidentally used post-release stream counts that wouldn't be available at prediction time, giving falsely optimistic accuracy numbers
