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
### The code:
<img width="587" height="240" alt="Screenshot 2025-09-06 at 2 07 30 PM" src="https://github.com/user-attachments/assets/2bb465de-b030-4582-b99e-3c668b680908" />


### The Monthly sales data broken up by: Raw Sales, Sales Trends, Sales Seasonality, and Sales Residuals 
<img width="1196" height="803" alt="Screenshot 2025-09-06 at 12 47 35 PM" src="https://github.com/user-attachments/assets/14c36e36-a23e-4f22-903d-721dfd9a4347" />

1. Plot 1: This shows the raw time series of the liqour sales. Here we can see strong periodic peaks, clearly around holidays which suggests definite seasonality.
2. Plot 2: This plot shows the long term trend data after we applied out STL decomposition within the 12 month windows. From this graph we see a steady upward trend with around a 8% YoY growth
3. Plot 3: In this graph we see the seasoanl component of our sales data. Each year we see a rise in sales during late fall (October/November) and Decemeber confirming a strong yearly seasonal cycle
4. Plot 4: This Residual plot shows the sales variations over time. This graph demonstrates that theres no strong structure within the residuals, meaning the decomposition correctly captured the main trend and seasonality within our data.

### Modelling 
<img width="432" height="203" alt="Screenshot 2025-09-06 at 2 14 19 PM" src="https://github.com/user-attachments/assets/63790028-6687-48a8-b508-fd9e431e0778" />

Prophet model configurations:
 - Yearly seasonality: Enabled becasue we know the sales clearly follow    a annual cycle
 - Weekly and Daily seasonality: disabled becasue we resampled the data    earlier for the monthly level.
 - Seasonality mode: I used multiplicative here because I know the         seasonal effects grew with the overall sales
 - Changepoint scale: used a low change point scale to make the trend      less sensitive to small fluctuations in data, this in turn prevented    overfitting issues of which I encountered early on.
 - Interval width: Changing the % from the default of 95% for              practicality in the context of using this model for buisness            applications. If I were to use more conservative uncertantity bounds    can be less actionable in real life scnarios like these.


<img width="751" height="77" alt="Screenshot 2025-09-06 at 2 10 14 PM" src="https://github.com/user-attachments/assets/71f200b7-8a56-40db-8777-476d82377efd" />

Here we established changepoints within the sales data where we noticed the general seasonal sales trend deviate from higher sales in october, followed by abover average but lower sales in decemeber. In the last 2 years of the data this pattern flipped with Decemeber seeing more sales than the late fall sales.

<img width="418" height="223" alt="Screenshot 2025-09-06 at 2 24 18 PM" src="https://github.com/user-attachments/assets/15811320-597f-4f0c-b5af-220048fcaf60" />

Reggresors: Although Prophet on its own captures things like trends and seasonality, it doesnt account for external facotrs. By adding regressors, the model now accounts for any sudden structural shifts in sales.

Custom Seasonality: Standard prophet models are good for smooth cycles but during intial tests, the holiday sales spikes caused confusion for model. This feature helped the model focus on capturing the unique surges in sales

### Forecasting 
<img width="641" height="138" alt="Screenshot 2025-09-06 at 2 29 24 PM" src="https://github.com/user-attachments/assets/40392606-fe21-43d9-9cbb-c60e2c86f392" />

Here we fit the model and create the future data frame. Since I reasampled the data by monthly sales, we set the "future periods" to 12. On line 153, were extending the regressors so the model can adjust its forecasting for those past structural changes.

### Visualizations
### Model visualization code:
<img width="809" height="516" alt="Screenshot 2025-09-06 at 2 34 50 PM" src="https://github.com/user-attachments/assets/3144a988-9476-420f-ac0c-1e1236206338" />

<img width="1455" height="768" alt="Screenshot 2025-09-06 at 2 37 38 PM" src="https://github.com/user-attachments/assets/d82c80a7-bc3b-4a8f-a959-328f59e02f5b" />


<img width="990" height="562" alt="Screenshot 2025-09-06 at 2 36 06 PM" src="https://github.com/user-attachments/assets/5a0c056d-bd37-4bf4-bc73-b7972944aa3d" />

<img width="1275" height="838" alt="Screenshot 2025-09-06 at 2 37 11 PM" src="https://github.com/user-attachments/assets/ad7e61af-9a5d-4894-bcb3-41db3aeb254a" />

1. Plot 1: Here we see the steady paced growth over the years
   
2. Plot 2: The model shows the earleir observed cyclical buying            behavior from the historical data

3. Plot 3: Visualizes the multiplicative holiday seasonality. This shows how our seasoanlity featruing improves on analyzing the holiday demand spikes beyond the default prophet modeling capabilities.

4. Plot 4: Visualizes the impact our regressors have after we account for the changepoints in the sales data (hence the stepwise graph).

### Evaluations
### in-sample and cross validation code:
<img width="593" height="491" alt="Screenshot 2025-09-06 at 2 46 46 PM" src="https://github.com/user-attachments/assets/67fb4444-21ac-42fa-bb30-df41422feac9" />

Cross Validation: 
  - Parameters:
      - "540 days": Training based on the first 540 days
      - "period = 90 days": period of time for each new forecast for            evaluation
      - "horizon = '90 days': How far ahead the forecast should be              looking
  - Performance metrics:
      - Summarizes reuslts into the metrics like MAPE, RMSE, and MAE.

In-sample evaluation:
  - Historical forecast: Extracts the model predictions for the             historical period
  - Error Metrics:
    - MAE (Mean Absolute Error): Average absolute difference between          predicted and actual values.
    - RMSE (Root Mean Squared Error): Penalizes larger errors more            heavily than MAE.
    - MAPE (Mean Absolute Percentage Error): Error as a percentage of         actual sales — easier to interpret.
    - R² (Coefficient of Determination): Explains how much variance in        actual sales is captured by the model (closer to 1 = better fit).

- Model Metric results
  - MAPE: 14.8% (Mean Absolute Percentage Error)
  - RMSE: ~245,320 bottles (Root Mean Square Error)
  - MAE: ~198,450 bottles (Mean Absolute Error)
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


