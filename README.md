# Modeling Song Popularity with Neural Networks and Bayesian Regression

**DS4420 Final Project**  
Siyuan Chen, Yifan Wang  
Instructor: Dr. Eric Gerber  
April 2026

## Overview

This project investigates whether audio features can predict song popularity on Spotify. We use a dataset of approximately 114,000 tracks and apply two machine learning methods: a manually implemented neural network (MLP) and a Bayesian linear regression model.

## Live App

[Launch Streamlit App](https://music-ds4420-project-aihzneyeo8qxbigxhgxm6y.streamlit.app/)

The app includes a project overview page and an interactive visualization of model results.

## Repository Structure

```
├── app.py                        # Streamlit application
├── model_prototype.ipynb         # Neural network implementation (Python/NumPy)
├── bayesian_regression.Rmd       # Bayesian regression implementation (R/rstanarm)
├── model_weights.npz             # Pre-trained MLP weights
├── dataset（1）.csv                   # Spotify Tracks Dataset
├── requirements.txt              # Python dependencies
└── README.md
```

## Methods

**Neural Network (Python)**  
A multilayer perceptron (5 → 64 → 32 → 1) implemented from scratch using NumPy only. Trained with mini-batch SGD, He initialization, and ReLU activations for 500 epochs.

**Bayesian Regression (R)**  
A Bayesian linear regression model fit using `stan_glm` with weakly informative priors. Posterior distributions estimated via HMC sampling (4 chains, 2000 iterations).

## Data

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — ~114,000 tracks with audio features and popularity scores.

Features used: `danceability`, `energy`, `tempo`, `loudness`, `valence`  
Target: `popularity` (0–100)

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Neural Network (MLP) | 21.52 | 17.59 | 0.071 |

Both models indicate that audio features alone have limited predictive power. The Bayesian model finds that loudness and danceability have positive effects on popularity, while energy and valence show negative relationships.

## Requirements

```
pip install -r requirements.txt
```

To run the app locally:
```
streamlit run app.py
```
