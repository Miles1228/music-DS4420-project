# Modeling Song Popularity: Neural vs Bayesian

## Overview
This project investigates whether audio features from Spotify tracks can be used to predict song popularity. We apply machine learning methods to analyze the relationship between musical characteristics and listener engagement.

The project compares two approaches:
- A Neural Network (implemented from scratch using NumPy) for prediction
- A Bayesian Regression model (implemented in R) for interpretability

The goal is to evaluate both predictive performance and the explanatory power of audio features.

---

## Dataset
The dataset consists of approximately 114,000 Spotify tracks, each containing audio features and a popularity score (0–100).

Selected features include:
- danceability
- energy
- tempo
- loudness
- valence

These variables are used as predictors to model song popularity.

---

## Methods
### Neural Network (Python)
- Architecture: 5 → 64 → 32 → 1  
- Activation: ReLU (hidden), Linear (output)  
- Optimization: Mini-batch SGD  
- Implemented manually using NumPy  

### Bayesian Regression (R)
- Linear model with priors  
- Posterior estimation using sampling  
- Provides interpretable coefficients and uncertainty estimates  

---

## Results
The neural network achieves moderate predictive performance but shows limited explanatory power (R² ≈ 0.07). Predictions tend to concentrate around the mean.

The Bayesian model reveals that features such as loudness and danceability have meaningful effects on popularity, while also quantifying uncertainty through credible intervals.

Both models suggest that audio features alone are insufficient to fully explain song popularity.

---

## Repository Structure


---

## Authors
- Siyuan Chen  
- Yifan Wang  
