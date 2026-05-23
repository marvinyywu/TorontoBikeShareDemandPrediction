# Toronto Bike Share Demand Prediction

## Problem
Bike share systems need to anticipate station-level demand so bikes can be rebalanced more effectively.

## Data
I used Bike Share Toronto ridership data from [City of Toronto Open Data](https://ckan0.cf.opendata.inter.prod-toronto.ca/en_AU/dataset/bike-share-toronto-ridership-data).

## Approach
I built a pipeline that ingests trip data, aggregates it into station-hour demand, creates time and lag features, trains ML models, and evaluates performance on future holdout data.

## Models
- Linear Regression

## Results
Reports MAE, RMSE, and R2.

## Engineering
- pandas preprocessing
- scikit-learn Pipeline
- time-based train/test split


## Limitations
The model uses historical trip data and may not fully capture sudden disruptions, station closures, weather shocks, or policy changes.

## How to Run

1. Download Bike Share Toronto ridership data for 2021 to 2026 from [City of Toronto Open Data](https://ckan0.cf.opendata.inter.prod-toronto.ca/en_AU/dataset/bike-share-toronto-ridership-data).
2. Structure the folders like so:
![Image of Dataset Folder Structure](images/Screenshot%202026-05-23%20194446.png)
1. run `pip install -r requirements.txt` 
2. run `python main.py`

