#Written by Didrik Wiig-Andersen
#Quantatech Securities
#24.09.2021


###########################################################################
#Import necessary dependencies
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.colors as clrs
import sys
##########################################################################


#Retrieve time specific data
today = date.today()
d1 = today.strftime('%Y/%m/%d')
d1_edit = d1.replace('/', '-')
d2_edit = d1_edit[-5:]
d2_edit_year = int(d1_edit[:4]) - 10
start = str(d2_edit_year)
z = start + '-' + d2_edit

#Asks for user input
print(d1)
ticker = input("TICKER: ")


#download OHLCV-data from yahoo
data = yf.download(ticker, start=z, end=d1_edit, auto_adjust = True)
data.index = pd.to_datetime(data.index)
data.to_csv(f"general_{ticker}.csv")


#Calculating signals for the ML model
data['future_returns'] = data['Close'].pct_change().shift(-1)
data['signal'] = np.where(data['future_returns'] > 0, 1, 0)

#Calculating features for the ML model
data['pct_change'] = data['Close'].pct_change()
data['pct_change_2'] = data['Close'].pct_change(2) #two days change
data['pct_change_5'] = data['Close'].pct_change(5) # 5 - weekday change

# Calculate single SMA
data['sma'] = data['Close'].rolling(window=int(6.5)).mean()

# Calculate the correlation between the sma and price of the security
data['corr'] = data['Close'].rolling(window=int(6.5)).corr(data['sma'])

# Calculate the single day volatility
data['volatility'] = data.rolling(
    int(6.5), min_periods=int(6.5))['pct_change'].std()*100

#Removing empty places in our dataframe
data.dropna(inplace=True)

#Copying the calculated signals
Y = data[['signal']].copy()

#Copying the calculated features
X = data[['pct_change', 'pct_change_2', 'pct_change_5','sma', 'corr', 'volatility']].copy()

#Checking if features are stationary
#Removing stationary features
def stationary(series):
    result = adfuller(series)
    if(result[1] < 0.05):
        return 'stationary'
    else:
        return 'not stationary'
for col in X.columns:
    if stationary(data[col]) == 'not stationary':
        X.drop(columns=[col], axis=1, inplace=True)

# Convert the features to a CSV file
X.to_csv(f"data/X_values_{ticker}.csv")

# Convert the targets to a CSV file
Y.to_csv(f"data/Y_values_{ticker}.csv")

#Alerts the user of where to find the files in the repo
print(f"""FEATURES AND TARGET VALUES WRITTEN ONTO
				 data/X_values_{ticker}.csv
				 data/Y_values_{ticker}.csv""")


### Machine Learning Section ###

#Splits the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split( X, Y, train_size=0.80, shuffle=False)

#Stores the testing and training signals anf features in four different files
X_train.to_csv(f"data/dataML/x_train_{ticker}.csv")
X_test.to_csv(f"data/dataML/x_test_{ticker}.csv")
y_train.to_csv(f"data/dataML/y_train_{ticker}.csv")
y_test.to_csv(f"data/dataML/y_test_{ticker}.csv")

#Alerts the user of where to find the files in the repo
print(f"""TRAINING AND TESTING DATA WRITTEN INTO
				 data/dataML/x_train_{ticker}.csv
				 data/dataML/x_test_{ticker}.csv
				 data/dataML/y_train_{ticker}.csv
				 data/dataML/y_test_{ticker}.csv""")


# Create the machine learning model
rf_model = RandomForestClassifier(n_estimators=3, max_features=3, max_depth=2, random_state=4)
rf_model.fit(X_train, y_train['signal'])

# Get a sample day of data from X_test
unseen_data_single_day = X_test.head(1)

# Get the prediction of a single day
single_day_prediction = rf_model.predict(unseen_data_single_day)

# Use the model and predict the values for the test data
y_pred = rf_model.predict(X_test)

# Save the predictions with the time index same as y_test
predicted_signal = pd.DataFrame(y_pred, index=y_test.index, columns=['signal'])

#write the predicted data onto a csv file
predicted_signal.to_csv(f"data/signals/predicted_{ticker}.csv")

#alert the user of where to find the file in the repo
print(f"""PREDICTED SIGNALS WRITTEN INTO
				 data/signals/predicted{ticker}.csv
				 """)

#Calculate and display a classification report
classification_report_data = classification_report(y_test, y_pred)
print(classification_report_data)

### Backtesting using the test data ###


strategy_data = pd.read_csv(f"data/signals/predicted_{ticker}.csv", index_col=0)
strategy_data['Close'] = pd.read_csv(f"general_{ticker}.csv", index_col=0).loc[strategy_data.index[0]:]['Close']
strategy_data.index = pd.to_datetime(strategy_data.index)

#Calculat the percentage change and strategy returns
strategy_data['pct_change'] = strategy_data['Close'].pct_change()
strategy_data['strategy_returns'] = strategy_data['signal'].shift(1) * \
strategy_data['pct_change']
strategy_data.dropna(inplace=True)

#Calculat the cummulative returns of the strategy
strategy_data['cumulative_returns'] = (1+strategy_data['strategy_returns']).cumprod()
strategy_data['cumulative_returns'].plot(figsize=(15, 7), color='green')
plt.title('Equity Curve', fontsize=14)
plt.ylabel('Cumulative returns')
plt.tight_layout()
cumulative_returns = (strategy_data['cumulative_returns'][-1] - 1)*100
print("The cumulative return is {0:.2f}%.".format(cumulative_returns))

#Calculat the annual returns of the strategy 
annualised_return = ((strategy_data['cumulative_returns'][-1]) **
                    (252*6.5/strategy_data.shape[0]) - 1) * 100
print("The annualised return is {0:.2f}%.".format(annualised_return))

# Calculate the annualised volatility of the strategy
annualised_volatility = strategy_data['strategy_returns'].std()*np.sqrt(
    252*6.5) * 100
print("The annualised volatility is {0:.2f}%.".format(annualised_volatility/4))

# Calculate the Sharpe ratio of the strategy
sharpe_ratio = round(strategy_data['strategy_returns'].mean() /
                     strategy_data['strategy_returns'].std() * np.sqrt(252*6.5), 2)
print("The Sharpe ratio is {0:.2f}.".format(sharpe_ratio))

#Calculate the maximum drawdown of the strategy and display it for the user
running_max = np.maximum.accumulate(
    strategy_data['cumulative_returns'].dropna())
running_max[running_max < 1] = 1
drawdown = ((strategy_data['cumulative_returns'])/running_max - 1) * 100
max_dd = drawdown.min()
print("The maximum drawdown is {0:.2f}%.".format(max_dd))
fig = plt.figure(figsize=(8, 3))
plt.plot(drawdown, color='red')
plt.fill_between(drawdown.index, drawdown.values, color='red')
plt.title('Strategy Drawdown', fontsize=14)
plt.ylabel('Drawdown(%)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.tight_layout()
plt.show()


