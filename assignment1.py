import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle

# Loading and preparing data
data = pd.read_csv('assignment_data_train.csv')

# Converting timestamp to datatime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Aggregate by timestamp in case of duplicates
data = data.groupby('Timestamp', as_index=False)['trips'].sum()

# Setting the timestamp as index
trips = data.set_index('Timestamp')['trips']

# Ensuring hourly frequency (filling missing hours with 0 trips)
trips = trips.resample('h').sum()

# Defining the model
model = ExponentialSmoothing(trips, trend='add', 
        seasonal='add', seasonal_periods=24*7, use_boxcox=True)

# Fitted model
modelFit = model.fit()

# Saving the fitted model
with open('taxi_model.pkl', 'wb') as f:
    pickle.dump(modelFit, f)

# Reloading the model to simulate future use
with open('taxi_model.pkl', 'rb') as f:
    fitModel = pickle.load(f)

# Loading the test data
test = pd.read_csv('assignment_data_test.csv')
test['Timestamp'] = pd.to_datetime(test['Timestamp'])

# No. of periods in the test set
n_periods = len(test)

# Forecasting for the test horizon
pred = fitModel.forecast(n_periods)

# Combining the test timestamps with predictions
results = pd.DataFrame({"TimeStamp": test['Timestamp'],
                            "Forecasted_trips": pred.values})

# Saving the results
results.to_csv("forecast_results", index=False)
        
