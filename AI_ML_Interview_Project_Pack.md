# AI/ML Interview Preparation Pack

Prepared for a Computer Engineering graduate preparing for an AI/ML engineering interview.

## 0. Executive Summary

The job description is not mainly asking for a large language model fine-tuning specialist. It is asking for someone who can work across the practical machine learning workflow:

- prepare real data,
- build a reliable data preprocessing pipeline,
- train and evaluate models in Python,
- write maintainable code,
- explain model performance and limitations,
- collaborate with data scientists, data engineers, and software engineers.

For this interview, the strongest preparation is not to memorize deep learning terminology. The strongest preparation is to complete one credible, reproducible project using real public data.

Recommended main project:

> Toronto Bike Share Demand Prediction: build an end-to-end machine learning pipeline that uses Toronto open data to predict bike-share demand by time and station.

This project is strong because it uses real Toronto data, has a clear business/operations problem, requires data cleaning and feature engineering, uses common ML frameworks, and can be explained in an interview without overclaiming.

## 1. Skills Required by the Job Description

### 1.1 Python for data and ML

He should be comfortable writing Python beyond basic syntax.

What to know:

- functions, modules, virtual environments,
- reading and writing CSV/ZIP files,
- working with file paths,
- using notebooks for exploration and `.py` files for repeatable code,
- basic error handling and debugging.

Target level:

He does not need to be a senior Python engineer. He does need to be able to run the project from scratch and explain every major file.

Practice target:

- Write `src/ingest.py` to load raw data.
- Write `src/preprocess.py` to clean and transform data.
- Write `src/train.py` to train models.
- Write `src/predict.py` to load a trained model and make predictions.

### 1.2 Data handling with pandas and NumPy

This is the first real skill gap for someone who only knows basic Python.

What to know:

- load CSV files with `pandas.read_csv`,
- inspect data with `.head()`, `.info()`, `.describe()`,
- handle missing values,
- parse dates and times,
- group by station, hour, day, and month,
- create new columns,
- remove duplicates and unreasonable records,
- merge datasets.

Why it matters:

The JD says "data pipelines" and "model-ready dataset preparation." This usually means taking messy real-world data and turning it into a clean table that a model can use.

Recommended learning:

- pandas Getting Started: https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html

### 1.3 Machine learning fundamentals

He needs practical understanding, not just formulas.

What to know:

- supervised learning,
- classification vs regression,
- train/test split,
- validation,
- overfitting,
- baseline model,
- feature engineering,
- model evaluation,
- data leakage.

For the proposed Bike Share project, the main task is regression:

> predict how many trips will start from a station during a future hour.

Recommended first models:

- Linear Regression,
- Random Forest Regressor,
- Gradient Boosting Regressor,
- XGBoost Regressor.

Recommended learning:

- scikit-learn Getting Started: https://scikit-learn.org/stable/getting_started.html
- scikit-learn ColumnTransformer example: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
- scikit-learn Common Pitfalls: https://scikit-learn.org/stable/common_pitfalls.html

### 1.4 scikit-learn pipelines

This is one of the highest-value skills for this interview.

What to know:

- `Pipeline`,
- `ColumnTransformer`,
- `OneHotEncoder`,
- `StandardScaler`,
- `SimpleImputer`,
- model evaluation with MAE, RMSE, R2.

Why it matters:

Many beginners clean data manually in notebooks and accidentally leak information from the test set into training. A pipeline shows engineering maturity because it packages preprocessing and model training into a repeatable workflow.

Interview explanation:

> I used scikit-learn Pipeline and ColumnTransformer so numeric, categorical, and time-based features are transformed consistently during training and prediction. This also reduces the chance of data leakage.

### 1.5 XGBoost or LightGBM

These are not mandatory for the first working version, but they are good portfolio upgrades.

What to know:

- gradient boosted decision trees,
- why tree models work well on tabular data,
- feature importance,
- comparing a strong model against a simple baseline.

Recommended learning:

- XGBoost documentation: https://xgboost.readthedocs.io/en/stable/

### 1.6 PyTorch and TensorFlow

Do not start here.

PyTorch and TensorFlow are important deep learning frameworks, but this JD is more strongly aligned with data preparation, model implementation, and ML engineering. For this interview, he should first show a clean scikit-learn pipeline. PyTorch can be a later extension.

When to learn PyTorch:

- after he can complete the Bike Share project,
- if the interviewer asks about neural networks,
- if he wants to build a second project involving images, text, or deep learning.

Recommended learning:

- PyTorch Learn the Basics: https://docs.pytorch.org/tutorials/beginner/basics/index.html

### 1.7 Software engineering practices

The JD also asks for maintainable code, code reviews, and team standards.

What to know:

- Git and GitHub,
- clean README,
- reproducible setup,
- requirements file,
- modular code,
- basic tests,
- clear commit history.

Minimum project engineering standard:

```text
project/
  README.md
  requirements.txt
  notebooks/
  src/
  data/
  models/
  reports/
```

Resume phrase after completing the project:

> Built an end-to-end machine learning pipeline using Python, pandas, scikit-learn, and XGBoost to forecast Toronto Bike Share hourly station demand from public open data; implemented data ingestion, preprocessing, feature engineering, model training, evaluation, and reproducible prediction scripts.

## 2. What the Toronto Open Data Website Is

Toronto Open Data is the City of Toronto's official public data portal. It is not a tutorial website and not a private company's project. It provides machine-readable public datasets that people can use to build analysis, research, apps, dashboards, and models.

Official explanation:

- City of Toronto says open data is information the City makes available for anyone to freely use, reuse, or analyze.
- The portal contains structured data such as CSV, XML, JSON, shapefiles, and APIs.
- The data can be used to generate insights, analyses, and web/mobile applications.

Sources:

- Introduction to open data: https://open.toronto.ca/docs/staff-guidance/introduction-to-open-data/
- What is Open Data: https://www.toronto.ca/city-government/data-research-maps/open-data/what-is-open-data/
- Open Data Licence: https://open.toronto.ca/open-data-licence/

Important interview wording:

> I used City of Toronto Open Data as the official data source. I did not use an official City prediction model. The model and pipeline were my own implementation.

Do not say:

> Toronto built this prediction system.

Do say:

> Toronto publishes the raw data. I used it to build my own ML pipeline and evaluate whether historical ridership patterns can predict demand.

## 3. Real Data and Reference Projects

### 3.1 Main dataset: Bike Share Toronto Ridership Data

Dataset:

- Bike Share Toronto Ridership Data
- Source: City of Toronto Open Data / CKAN
- Licence: Open Government Licence - Toronto

The dataset contains anonymized trip data, including:

- trip ID,
- trip duration,
- trip start station ID,
- trip start time,
- trip start station location,
- trip end station ID,
- trip end time,
- trip end station location,
- bike ID,
- user type.

Dataset page:

- https://ckan0.cf.opendata.inter.prod-toronto.ca/en_AU/dataset/bike-share-toronto-ridership-data

Additional catalogue mirror:

- https://data.urbandatacentre.ca/catalogue/city-toronto-bike-share-toronto-ridership-data

Why it is suitable:

- It is real operational data.
- It has timestamps and locations.
- It requires cleaning and aggregation.
- It supports a realistic forecasting problem.
- It connects naturally to transportation operations: station balancing, bike availability, maintenance planning, and commuter demand.

### 3.2 Reference project: Toronto Bike Share 2021 Analysis

GitHub:

- https://github.com/dailyLi/toronto_bike_share

What it does:

- analyzes Bike Share Toronto 2021 data,
- creates interactive visualization,
- uses Bike Share ridership data, station profile JSON, and daily weather data,
- uses Python, pandas, and Plotly.

How to use it:

- Use it as a reference for data cleaning and visualization.
- Do not copy the project.
- Notice how it structures notebooks and uses station/weather data.

### 3.3 Reference project: Open Data Toronto Bike Share Python article

Article:

- https://medium.com/open-data-toronto/exploring-toronto-bike-share-ridership-using-python-5c0e79fad442

Key lesson:

The article emphasizes a real-world truth: data cleaning can take most of the time in an analysis. It also discusses issues like inconsistent dates, station IDs/names, and time zones.

How to use it:

- Read it before coding.
- Extract the cleaning problems as interview talking points.
- Do not just reproduce the plots.

### 3.4 Reference project: School of Cities Bike Share analysis

E-bike analysis:

- https://schoolofcities.github.io/bike-share-toronto/efit-analysis

Trip mapping:

- https://schoolofcities.github.io/bike-share-toronto/trips-062024

What it does:

- analyzes Bike Share trip behavior,
- compares electric and classic bikes,
- estimates routes using network data,
- uses Python and D3 visualization.

How to use it:

- Use as inspiration for richer project ideas.
- Do not attempt this full geospatial complexity first.
- Consider it an advanced extension after the basic ML pipeline works.

### 3.5 Backup dataset: TTC delay data

TTC delay data is interesting, but currently the Toronto Open Data page surfaced in search appears as retired or partially incomplete. It can still be useful as a reference direction, but Bike Share is a safer main project.

Possible TTC project:

> Predict whether a reported transit delay will become a long delay based on line, station, time, and delay reason.

Use TTC only as a second project or extension, not the main interview project.

## 4. Proposed Interview Project

### 4.1 Project title

Toronto Bike Share Demand Prediction

### 4.2 One-sentence pitch

> I built an end-to-end machine learning pipeline that uses Toronto Bike Share open data to forecast hourly bike demand by station, using Python, pandas, scikit-learn, and XGBoost.

### 4.3 Business problem

Bike share systems often face two operational problems:

- some stations run out of bikes,
- some stations become full and cannot accept returns.

If the operator can forecast demand by station and time, it can better plan rebalancing, maintenance, and station capacity.

This is a real operations problem, not just a classroom prediction task.

### 4.4 Machine learning problem

Recommended task:

> Predict the number of trips that will start from a given station during a given hour.

Input features:

- station ID,
- hour of day,
- day of week,
- month,
- weekend flag,
- rush hour flag,
- previous hour demand,
- previous day same-hour demand,
- rolling average demand,
- weather features if available.

Target:

- trip count for that station and hour.

Model type:

- regression.

Evaluation metrics:

- MAE: average absolute error in predicted trip count,
- RMSE: penalizes large mistakes,
- R2: rough explanatory power,
- baseline comparison.

### 4.5 Baseline model

He must build a baseline first.

Baseline examples:

- predict the historical average demand for that station and hour,
- predict the previous week's same-hour demand,
- predict the previous day's same-hour demand.

Why this matters:

A model is not useful just because it runs. It is useful only if it beats a simple baseline.

Interview wording:

> I first built a baseline so I could measure whether the ML model actually added value. The final model needed to beat the naive historical-average forecast.

### 4.6 Feature engineering

Required features:

- `hour`: extracted from start time,
- `day_of_week`: Monday to Sunday,
- `month`: captures seasonality,
- `is_weekend`: weekend behavior differs from weekday commuting,
- `is_rush_hour`: morning/evening commute pattern,
- `station_id`: station-level location behavior,
- `lag_1h_demand`: demand in the previous hour,
- `lag_24h_demand`: same hour yesterday,
- `rolling_7d_avg`: average demand in recent comparable periods.

Optional features:

- temperature,
- precipitation,
- snow/rain flag,
- holidays,
- station capacity,
- nearby transit station or university area.

### 4.7 Data pipeline

Pipeline stages:

1. Ingest raw Bike Share ridership data.
2. Standardize column names.
3. Parse timestamps.
4. Remove invalid records.
5. Aggregate trips into station-hour demand.
6. Add time features.
7. Add lag and rolling features.
8. Split train/test by time, not random split.
9. Train baseline and ML models.
10. Evaluate and save model.
11. Provide prediction script or API.

Important: time-based split

Do not randomly split time-series-like operational data. Train on earlier months and test on later months. This is more realistic because the model should predict the future from the past.

Interview wording:

> Because this is a forecasting problem, I used a time-based split instead of random train/test split. That better simulates how the model would be used in production.

### 4.8 Model choices

Version 1:

- Linear Regression,
- Random Forest Regressor.

Version 2:

- XGBoost Regressor,
- LightGBM Regressor if time allows.

Do not use a neural network first. A neural network is not needed to prove competence for this JD.

### 4.9 Repository structure

```text
toronto-bike-share-demand/
  README.md
  requirements.txt
  .gitignore
  notebooks/
    01_exploratory_analysis.ipynb
    02_model_experiments.ipynb
  src/
    config.py
    ingest.py
    preprocess.py
    features.py
    train.py
    evaluate.py
    predict.py
  data/
    raw/
    processed/
  models/
  reports/
    figures/
    model_metrics.md
  tests/
    test_features.py
```

### 4.10 Minimum deliverables

He should not go to the interview with only a notebook.

Minimum:

- GitHub repository,
- clean README,
- one exploratory notebook,
- reusable Python scripts,
- trained model file,
- metrics table,
- at least one visualization,
- short project summary.

Better:

- Streamlit dashboard or FastAPI endpoint,
- simple unit tests,
- GitHub Actions test workflow,
- comparison of baseline vs ML models,
- feature importance plot.

### 4.11 README outline

```markdown
# Toronto Bike Share Demand Prediction

## Problem
Bike share systems need to anticipate station-level demand so bikes can be rebalanced more effectively.

## Data
I used Bike Share Toronto ridership data from City of Toronto Open Data.

## Approach
I built a pipeline that ingests trip data, aggregates it into station-hour demand, creates time and lag features, trains baseline and ML models, and evaluates performance on future holdout data.

## Models
- Historical average baseline
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

## Results
Report MAE, RMSE, and R2.

## Engineering
- pandas preprocessing
- scikit-learn Pipeline
- time-based train/test split
- model saved with joblib
- reproducible scripts

## Limitations
The model uses historical trip data and may not fully capture sudden disruptions, station closures, weather shocks, or policy changes.

## How to Run
...
```

## 5. Four-Week Work Plan

If the interview is very soon, use the compressed 10-day version after this section. If there is enough time, use the four-week version.

### Week 1: Learn the data and pandas

Goal:

- understand the dataset,
- clean a one-month sample,
- aggregate trips by station and hour.

Deliverables:

- `01_exploratory_analysis.ipynb`,
- data dictionary in README,
- first cleaned dataset.

Must be able to explain:

- what each column means,
- what records were removed,
- what station-hour demand means,
- why cleaning real-world data is hard.

### Week 2: Build the first ML pipeline

Goal:

- build baseline,
- train Linear Regression and Random Forest,
- evaluate with MAE/RMSE/R2.

Deliverables:

- `src/preprocess.py`,
- `src/features.py`,
- `src/train.py`,
- `reports/model_metrics.md`.

Must be able to explain:

- train/test split,
- why time split is better than random split,
- what MAE means in business language,
- why baseline comparison matters.

### Week 3: Improve model and engineering

Goal:

- add XGBoost,
- improve features,
- save the model,
- add prediction script.

Deliverables:

- `src/predict.py`,
- saved model in `models/`,
- feature importance chart,
- updated README.

Must be able to explain:

- why XGBoost may work well on tabular data,
- which features matter,
- where the model fails,
- how to avoid data leakage.

### Week 4: Interview polish

Goal:

- make the project easy to review,
- practice interview explanation,
- prepare resume bullets.

Deliverables:

- final GitHub repo,
- 2-minute project pitch,
- 5-minute technical walkthrough,
- resume bullet points,
- one-page project summary.

Optional:

- Streamlit app,
- FastAPI endpoint,
- simple tests,
- GitHub Actions.

## 5A. Compressed 10-Day Version

Use this if the interview is close and the goal is to become credible fast.

### Days 1-2: pandas and data understanding

Tasks:

- download one year or one quarter of Bike Share data,
- load it with pandas,
- understand columns,
- create station-hour demand table,
- write down all cleaning decisions.

Output:

- one notebook,
- one cleaned CSV,
- short data dictionary.

### Days 3-4: first model

Tasks:

- create time features,
- create train/test split by date,
- build baseline,
- train Linear Regression and Random Forest,
- evaluate with MAE/RMSE/R2.

Output:

- metrics table,
- baseline comparison,
- first README draft.

### Days 5-6: pipeline and feature engineering

Tasks:

- move notebook logic into `src/preprocess.py`, `src/features.py`, and `src/train.py`,
- add lag features,
- add rolling average features,
- make the model reproducible from command line.

Output:

- reusable scripts,
- model file saved with `joblib`,
- feature importance chart.

### Days 7-8: XGBoost and project polish

Tasks:

- add XGBoost,
- compare models,
- add charts,
- clean repo structure,
- improve README.

Output:

- final model comparison,
- complete GitHub repository.

### Days 9-10: interview rehearsal

Tasks:

- practice 30-second pitch,
- practice 2-minute technical explanation,
- answer validation questions in Section 8,
- prepare resume bullets.

Output:

- final project summary,
- resume-ready bullets,
- answers to likely interviewer questions.

If time runs out, skip Streamlit/FastAPI. A clean reproducible ML pipeline is more important than a thin app wrapper.

## 6. Interview Talk Track

### 30-second version

> I built a Toronto Bike Share demand prediction project using City of Toronto open data. The goal was to forecast hourly station-level bike demand. I created a data pipeline in Python to clean raw trip data, aggregate it into station-hour demand, build time and lag features, and train regression models using scikit-learn and XGBoost. I evaluated the models using a time-based train/test split and compared them against a historical-average baseline.

### 2-minute version

> The project started with raw Bike Share trip data from Toronto Open Data. Each record represented an anonymized trip, including start time, end time, start station, end station, trip duration, and user type. I transformed this raw trip-level data into a model-ready station-hour dataset. That required timestamp parsing, station cleanup, aggregation, and feature engineering.
>
> I created features such as hour of day, day of week, month, weekend flag, rush-hour flag, previous-hour demand, previous-day same-hour demand, and rolling averages. Because this is a forecasting problem, I split the data by time instead of using a random split.
>
> I trained a historical-average baseline, Linear Regression, Random Forest, and XGBoost. I evaluated performance using MAE, RMSE, and R2. The most important part was not just training a model, but making the pipeline reproducible and being honest about limitations, such as weather shocks, station changes, and special events.

### Strong interview points

- "I started with a baseline."
- "I used time-based validation."
- "I avoided data leakage."
- "I turned raw trip records into station-hour demand."
- "I used a reproducible pipeline instead of only notebook code."
- "I can explain model errors and limitations."

### Things not to claim

- Do not claim this is an official Toronto model.
- Do not claim the model predicts all future demand perfectly.
- Do not claim deep learning was necessary.
- Do not claim TensorFlow/PyTorch experience unless he actually used them.

## 7. Resume Skill Mapping

### Skill section after completing the project

```text
Languages: Python, SQL
Data/ML: pandas, NumPy, scikit-learn, XGBoost, matplotlib/Plotly
ML Concepts: supervised learning, regression, feature engineering, model evaluation, train/test validation, data leakage prevention
Engineering: Git, GitHub, virtual environments, reproducible scripts, basic testing
Optional: FastAPI or Streamlit, joblib model serialization
```

### Project bullet examples

Use only bullets that are true.

```text
- Built an end-to-end ML pipeline using Python, pandas, scikit-learn, and XGBoost to forecast hourly station-level Bike Share Toronto demand from public open data.
- Engineered time-series features including hour-of-day, weekday, rush-hour flags, lag demand, and rolling averages from raw trip-level records.
- Evaluated baseline, linear, tree-based, and gradient boosting models using time-based validation and regression metrics including MAE, RMSE, and R2.
- Packaged preprocessing, training, evaluation, and prediction into reusable Python scripts with a reproducible GitHub repository and clear README documentation.
```

## 8. How to Validate Whether He Really Understands It

Ask him these questions.

### Data questions

1. What does one raw row represent?
2. What does one model training row represent after preprocessing?
3. Why did you aggregate trips by station and hour?
4. What records did you remove and why?
5. Did any station names or station IDs need cleaning?

### Feature questions

1. Why is hour of day useful?
2. Why is weekend behavior different?
3. What is a lag feature?
4. What is a rolling average feature?
5. Which features were most useful?

### ML questions

1. Is this classification or regression?
2. What is the baseline?
3. What is MAE?
4. Why not use accuracy?
5. Why use time-based split?
6. What is data leakage?
7. Why might XGBoost beat Linear Regression?

### Engineering questions

1. How do I run your project from a fresh computer?
2. Which file trains the model?
3. Which file makes predictions?
4. Where are model metrics stored?
5. What would you change if this had to run daily?

If he cannot answer these, he is not interview-ready yet.

## 9. Reference Pack

### Official data and documentation

- Toronto Open Data introduction: https://open.toronto.ca/docs/staff-guidance/introduction-to-open-data/
- City of Toronto "What is Open Data": https://www.toronto.ca/city-government/data-research-maps/open-data/what-is-open-data/
- Open Data Licence: https://open.toronto.ca/open-data-licence/
- Bike Share Toronto Ridership Data: https://ckan0.cf.opendata.inter.prod-toronto.ca/en_AU/dataset/bike-share-toronto-ridership-data
- Bike Share catalogue mirror: https://data.urbandatacentre.ca/catalogue/city-toronto-bike-share-toronto-ridership-data

### Python and ML foundations

- pandas Getting Started: https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html
- scikit-learn Getting Started: https://scikit-learn.org/stable/getting_started.html
- scikit-learn ColumnTransformer example: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
- scikit-learn Common Pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
- XGBoost documentation: https://xgboost.readthedocs.io/en/stable/
- PyTorch Learn the Basics: https://docs.pytorch.org/tutorials/beginner/basics/index.html

### Reference projects to study

- Toronto Bike Share 2021 Analysis and Interactive Visualization: https://github.com/dailyLi/toronto_bike_share
- Open Data Toronto Bike Share Python article: https://medium.com/open-data-toronto/exploring-toronto-bike-share-ridership-using-python-5c0e79fad442
- School of Cities e-bike analysis: https://schoolofcities.github.io/bike-share-toronto/efit-analysis
- School of Cities trip mapping: https://schoolofcities.github.io/bike-share-toronto/trips-062024
- End-to-end bike sharing demand prediction repo: https://github.com/Pratik94229/Bike-Sharing-Demand-Prediction-End-to-End-Project
- Bike Sharing Demand Prediction example repo: https://github.com/Apaulgithub/Bike_Sharing_Demand_Prediction
- Kaggle-style bike sharing prediction repo: https://github.com/owenpb/Kaggle-Bike-Sharing-Prediction

### YouTube-style project references

Use these as workflow references, not as code to copy.

- Krish Naik, "End to End Machine Learning Project Implementation with Dockers, GitHub Actions and Deployment" via YouTube/Class Central: https://www.classcentral.com/course/youtube-end-to-end-machine-learning-project-implementation-with-dockers-github-actions-and-deployment-120579
- Search YouTube for: "scikit-learn Pipeline ColumnTransformer tutorial"
- Search YouTube for: "end to end machine learning project scikit-learn pipeline"
- Search YouTube for: "bike sharing demand prediction Python scikit-learn"
- Search YouTube for: "Kaggle bike sharing demand prediction Python"

When watching videos, focus on:

- project structure,
- preprocessing pipelines,
- model evaluation,
- saving/loading models,
- deployment with Streamlit/FastAPI,
- not just notebook accuracy.

## 10. Final Recommendation

Do not make TensorFlow or PyTorch the starting point.

The best interview preparation path is:

```text
pandas -> scikit-learn -> Bike Share demand project -> XGBoost -> optional Streamlit/FastAPI -> optional PyTorch later
```

The project should be judged by whether he can explain:

- what the data means,
- how raw data becomes model-ready data,
- why he chose each feature,
- how he evaluated the model,
- what the model cannot do,
- how the code is organized so another engineer can run it.

If he can complete and explain this project well, it directly addresses the highlighted JD requirements: hands-on ML model implementation, Python, common ML libraries, data pipelines, preprocessing, feature engineering, and model-ready dataset preparation.
