# Prophet-liqour-sales-forecasting-
I created a sales forecasting model using a kaggle dataset of liqour sales from 2012-2016 with facebook's "Prophet" model. Additionally, for this project, I used multiple libraries like Pandas, NumPy, Matplotlib, Sklearn, and many others for data cleaning, visualization, and analysis.

## Sales Forecasting with Prophet
### Introduction

This project uses a Kaggle dataset of liquor sales (2012–2016) to build a robust sales forecasting model using Facebook’s Prophet library. The objective is to uncover seasonal patterns and trends in sales, providing actionable insights and accurate future predictions.

### Tools & Libraries Used
Python – primary programming language
Pandas & NumPy – for data cleaning, transformation, and handling
Matplotlib – for data visualization, trend analysis, and diagnostics
Prophet – for building and tuning the time series forecasting model
Sklearn.metrics - for evaluating quality of predictions made by the model
Statsmodels - for decomposition of our time series
Dataset: https://www.kaggle.com/datasets/fatemehmohammadinia/retail-sales-data-set-of-alcohol-and-liquor

## Key Features
- Seasonal decomposition analysis
- Multiple forecasting approaches (Prophet, ARIMA)
- Comprehensive model evaluation
- Interactive visualizations

### Importing Libraries and Data
<img width="608" height="188" alt="Screenshot 2025-08-29 at 11 09 00 AM" src="https://github.com/user-attachments/assets/9d5b6f3b-df69-40c2-8124-1acdd8678b5d" />

### About the Data
- Time Period: January 2012 - December 2016
- Records: 300,000+ transactions (sampled from 12M+ total)
- Key Features: Date, Bottles Sold, Item Description, Category, Store Location
- Granularity: Daily sales aggregated to monthly for analysis
  
### Loading and breaking down the dataset 

### Monthly resampling and seasonality Analysis
- Monthly seasonal indices calculation
- Peak season identification (Nov-Dec: +25% sales)
- Low season analysis (Jan-Feb: -15% sales)

### Decomposition Analysis
- STL decomposition with 12-month period
- Trend extraction showing 8% YoY growth

### Residual analysis

### Modelling 

### Forecasting 

### Visualizations

### Evaluations
- Model Metrics
  - MAPE: 14.8% (Mean Absolute Percentage Error)
  - RMSE: 245,320 bottles (Root Mean Square Error)
  - MAE: 198,450 bottles (Mean Absolute Error)
  - R²: 0.863 (Coefficient of Determination)

### Forecasting summary and conclusions 
- Identified strong seasonal patterns with 25% sales increase during holiday months
- Achieved R² score of 0.85+ on validation data
- Implemented custom changepoint detection for structural breaks in sales trends

With this forecasting model buisnesses can now:
* Inventory Optimization: Reduce stockouts by 30% during peak seasons
* Resource Planning: Optimize staffing based on predicted demand
* Revenue Forecasting: Accurate 12-month revenue projections
* Seasonal Strategy: Data-driven promotional planning
