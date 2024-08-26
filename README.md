# The Impact of the Weather Forecast Model on Improving AI-based Power Generation Predictions through BiLSTM Networks

## Abstract
This study aims to comprehensively analyze five weather forecasting models obtained from the Open-Meteo historical data repository, with a specific emphasis on evaluating its impact in predicting wind power generation. Given the increasing focus on renewable energy, namely wind power, accurate weather forecasting plays a crucial role in optimizing energy generation and ensuring the stability of the power system. The analysis conducted in this study incorporates a range of models, namely ICOsahedral Nonhydrostatic (ICON),  Global Environmental Multiscale Model (GEM Global), Meteo France, Global Forecast System (GSF Global), and the Best Match technique. The Best Match approach is a distinctive solution provided by the provider that combines the data from all available models to generate the most precise forecast for a particular area. The performance of these models was evaluated using various important metrics, including the mean squared error, the root mean squared error, the mean absolute error, the mean absolute percentage error, the coefficient of determination, and the normalized mean absolute error. The weather forecast model output was used as an essential input for the power generation prediction models during the evaluation process. This method was confirmed by comparing the predictions of these models with actual data on wind power generation. The ICON model, for example, outscored others with a mean squared error of 3.0852 and a root mean squared error of 1.7565, a tiny but essential improvement over Best Match, which had a mean squared error of 3.099 and root mean squared error of 1.7604. GEM Global and Gsf Global showed more dramatic changes, with Mean Squared Error (MSEs) of 4.0346 and 4.0973, respectively, indicating a loss in prediction accuracy of around 24\% to 31\% compared to ICON. Our findings reveal significant disparities in the precision of the various models used and certain models exhibit significantly higher predictive precision.

## Installation

pip install openmeteo-requests
pip install requests-cache retry-requests numpy pandas

## Usage

Steps to Reproduce the Results
### Download Weather Forecast Data:

Obtain meteorological data from the relevant weather service or API. Ensure that the data includes necessary variables such as wind speed, temperature, humidity, etc., for the period of interest.

### Download Wind Farm Data:
Collect wind farm data, which should include power generation data corresponding to the same time period as the meteorological data. Ensure that this data is in a compatible format for analysis.

### Join the Data:
Merge the meteorological data with the wind farm data based on timestamps. The combined dataset should align the weather conditions with the corresponding power generation outputs.
Save the merged dataset in the following format: raw_data.mat. This file will be used as the input for further analysis.

### Prepare the raw_data.mat File:
Ensure that `raw_data.mat` contains all the necessary variables (e.g., weather features and power generation targets) required for model training and evaluation.
Place the raw_data.mat file in the appropriate directory within the repository where main.m can access it.

### Run the Main Script:
Execute the `main.m` script multiple times, each time specifying a different model and a unique subdirectory name to save the results.
This will save the results in different folders, allowing you to organize and compare the outputs efficiently.

### Compare Results:

Use the `compare.m` script to compare the performance of the different models.
This script will analyze the results stored in the various subdirectories and generate comparative metrics or visualizations to evaluate which model performs best.

### Request Real Data for Comparison

If you would like to request access to the real data used for comparison, please [click here](mailto:info@inionsoftware.com?subject=Request%20for%20Real%20Data%20for%20Power%20Generation%20Prediction%20Project) to send a request via email.

## Contact
m.jankauskas@vilniustech.lt
