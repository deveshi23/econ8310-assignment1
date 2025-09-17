import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle

# Loading the data
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
df = data
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
df = df.asfreq('h')

# Dependent variable to forecast
y_train = df['trips']

# Defining forecasting model function
def model(series):
    # Assuming that weekly seasonality is 168 hours
    return ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=168)

# Fitting the model
modelFit = model(y_train).fit()

# Predicting for the next 744 hours 
pred = modelFit.forecast(steps=744)

# Converting the predictions to vectory (Numpy array)
pred_vector = pred.values

# Saving the fitted model for later use
with open("modelFit.pkl", "wb") as f:
    pickle.dump(modelFit, f)

# Showing the first few predictions
print(pred.head())