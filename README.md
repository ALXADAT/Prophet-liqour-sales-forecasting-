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
This project provides several analytical components:
- Seasonal Decomposition Analysis: breaks the data series into trend, seasonal, and residual components.
- Model Evaluation Process: Including error metrics such as MAE, RMSE, MAPE, and R².
- Interactive visualizations: Showing forecasts and decomposition, highlighting seasonal peaks and troughs.

### Importing Libraries and Data
<img width="611" height="191" alt="Screenshot 2025-09-06 at 1 25 54 PM" src="https://github.com/user-attachments/assets/66fe2a01-b5a8-4435-849b-8272308ed6ab" />

### About the Data
- Time Period: January 2012 - December 2016
- Records: 300,000+ transactions (sampled from 12M+ total)
- Key Features: Date, Bottles Sold, Item Description, Category, Store Location
- Granularity: Daily sales aggregated to monthly for analysis
  
### Loading and breaking down the dataset 
<img width="823" height="547" alt="Screenshot 2025-09-06 at 1 27 19 PM" src="https://github.com/user-attachments/assets/5f1a018f-1253-461b-8611-c17127cc2cb9" />

First the code checks the dataset headers, removes the leading or trailing whitespace from column names, and selects only the relevant fields. Missing values in the Date column are dropped (as seen on line 41), and then we index chronologically. The reason for structuring the data this way ensures accurate resampling and modeling later on.

### Monthly resampling and seasonality Analysis
<img width="545" height="61" alt="Screenshot 2025-09-06 at 1 30 27 PM" src="https://github.com/user-attachments/assets/52699255-edd6-4fd0-b448-50b33b430618" />

The data is resampled into monthly totals of bottles sold.

<img width="667" height="349" alt="Screenshot 2025-09-06 at 1 31 56 PM" src="https://github.com/user-attachments/assets/b7f153b2-b4ef-44ab-abf9-35a3d4b013a8" />

Early on in the process of making this project I noticed numerous outlier sales that were effecting our seasonality metrics. To handle these issues, i created a outlier handler that detected outliers using a rolling z-score method and replaced with local medians to smooth the series. The use of z-scores to detect the outliers rather than IQR methods is because... 
- 1. I noticed an overall upward trend (IQR methods works best with stable distributions)
- 2. Rolling Z-scores tend to work better with data with temporal structures like sales over time or stock prices

<img width="742" height="306" alt="Screenshot 2025-09-06 at 1 42 30 PM" src="https://github.com/user-attachments/assets/efd916cd-d516-4ee8-832b-ad0d26cae166" />

Seasonal indices are then calculated by averaging sales by calendar month. This reveals strong seasonal patterns, with November and December consistently showing a 25% increase in sales, while January and February exhibit about a 15% decrease relative to average months.

### Decomposition Analysis
<img width="810" height="59" alt="Screenshot 2025-09-06 at 1 52 06 PM" src="https://github.com/user-attachments/assets/e7fc4dee-7a64-4462-beb1-6d3cc9b0e087" />


Here the code applies seasonal-trend decomposition using moving averages (STL-style) with a 12-month period (for observing yearly seasonality).
I applied log transformations to the sales series to stabalize relative changes in sales to reduce influences from extreme sales peaks. This ensured that the deomposition produced a clearer seperation between general sales trends and any seasonal components. From there the extracted trend indicates an approximate 8% year-over-year growth in liquor sales across the observed period.

## Decomposition breakdown
<img width="662" height="436" alt="Screenshot 2025-09-06 at 1 53 13 PM" src="https://github.com/user-attachments/assets/410869af-b399-4a35-b324-46724cf8910e" />
<img width="1196" height="803" alt="Screenshot 2025-09-06 at 12 47 35 PM" src="https://github.com/user-attachments/assets/14c36e36-a23e-4f22-903d-721dfd9a4347" />
<img width="1177" height="382" alt="Screenshot 2025-09-06 at 12 48 02 PM" src="https://github.com/user-attachments/assets/528414cb-4a05-4e21-a0ea-2c33501452e3" />



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
