#Import libraries
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from web3 import Web3
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
from funcs.streamlit_functions import check_stationarity
st.set_option('deprecation.showPyplotGlobalUse', False)
import itertools
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hvplot.pandas  # Import HvPlot for Pandas
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import dim, opts
from bokeh.plotting import show  # Import show function from Bokeh
from statsmodels.tsa.arima.model import ARIMA
import pickle
# initialize bokeh
hv.extension('bokeh')

# Explain 
### Initialize the Streamlit Page
st.title("Gas Price Forecasting")
st.header("This page forecasts the gas price for the next 365 days.")
st.image("Images/Ethereum-Hand.png", width=350)
# Create a divider to separate the title and the rest of the content
st.markdown("***")
# Explain How the ARIMA Model Works, Ethereum Gas prices, and the data source:
# Add links to relevant resources
with st.expander("Background:"):
    st.subheader("How the ARIMA Model Works")
    st.write("https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/")
    st.markdown("The ARIMA model is a statistical model that uses time series data to predict future values. It is a combination of two other models: the autoregressive model (AR) and the moving average model (MA). The AR model uses the dependent relationship between an observation and some number of lagged observations. The MA model uses the dependency between an observation and a residual error from a moving average model applied to lagged observations. The I (for integrated) indicates that the data values have been replaced with the difference between their values and the previous values (and this differencing process may have been performed more than once).")
    st.markdown("The ARIMA model is defined by three parameters: p, d, and q. The p parameter is the number of lag observations included in the model, also called the lag order. The d parameter is the number of times that the raw observations are differenced, also called the degree of differencing. The q parameter is the size of the moving average window, also called the order of moving average.")
    st.markdown("The ARIMA model can be used to forecast the gas price for the next 365 days. The model will be trained on the gas price data from the last 365 days. The model will then be used to predict the gas price for the next 365 days.")
    st.subheader("Ethereum Gas Prices")
    st.write("https://ethereum.org/en/developers/docs/gas/")
    st.markdown("Gas is the fee paid to miners to include a transaction in a block. Gas prices are denoted in Gwei, which is a denomination of Ether (ETH). 1 ETH = 1,000,000,000 Gwei. The gas price is determined by the miners. The higher the gas price, the faster the transaction will be processed. The gas price is determined by the miners. The higher the gas price, the faster the transaction will be processed.")
    st.subheader("Data Source")
    st.write("https://etherscan.io/chart/gasprice")
    st.markdown("The gas price data is obtained from the Ethereum blockchain. The data is obtained using the Alchemy API. The Alchemy API is a blockchain developer platform that provides a suite of tools and infrastructure to help developers build great products on the Ethereum network. The Alchemy API is used to connect to the Ethereum blockchain and fetch the gas price data.")
    # Create a divider to separate the title and the rest of the content
st.markdown("***")

with st.expander("Part 1:"):
    st.subheader("Data Collection")
    st.warning("Depending on the selected timeframe, wait times can reach 20 Minutes and more.")



    # if st.button("Load Live Data") and (not pd.isnull(end_date)) and (not pd.isnull(start_date)):
    #     st.session_state['start_date'] = start_date
    #     st.session_state['end_date'] = end_date
    #     # Load .env file
    #     load_dotenv()

    #     # Load API key from .env file
    #     api_key = os.getenv("ALCHEMY_API_KEY")

    #     # Define the Alchemy API endpoint
    #     api_endpoint = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"

    #     # Set up Web3 connection
    #     w3 = Web3(Web3.HTTPProvider(api_endpoint))

    #     # Check if the connection is successful
    #     if w3.is_connected():
    #         st.write("Connected to Ethereum node.")
    #     else:
    #         st.write("Connection failed!")
        
    #     # Display loading message
    #     loading_message = st.text('Loading...')
        
    #     # Simulate some time-consuming process
    #     time.sleep(5)

    #     # Define the block time(in seconds)
    #     avg_block_time = 13  
    #     st.write(f"Average block time: {avg_block_time} seconds")

    #     # Calculate the estimated number of blocks per hourn(3600 seconds in an hour)
    #     blocks_per_hour = int(3600 / avg_block_time)
    #     st.write(f"Estimated blocks per hour: {blocks_per_hour}")

    #     # Calculate the total hours for the given time period
    #     total_hours = int((end_date - start_date).total_seconds() / 3600)

    #     # Calculate the total number of blocks for the given time period
    #     total_blocks = total_hours * blocks_per_hour
    #     st.write(f"Total number of hourly blocks for the given time period: {total_blocks}")

    #     # Get the latest block number
    #     latest_block = w3.eth.block_number
    #     st.write(latest_block)

    #     # Calculate the start block
    #     start_block = latest_block - total_blocks
    #     start_block

    #     # Initialize an empty DataFrame to store the gas prices
    #     gas_price_data = pd.DataFrame(columns=['timestamp', 'gas_price'])

    #     # Loop through the blocks, show status bar
    #     for block_number in tqdm(range(start_block, latest_block, blocks_per_hour),
    #                             desc="Fetching gas prices data"): 
            
    #         # Get the block details
    #         block = w3.eth.get_block(block_number)

    #         # Convert the gas price from hex to integer
    #         gas_price = int(str(block['baseFeePerGas']), 16)

    #         # Convert timestamp to datetime and create a DataFrame with the block timestamp and gas price
    #         block_data = pd.DataFrame({
    #             'timestamp': [datetime.fromtimestamp(block['timestamp'])],
    #             'gas_price': [gas_price]
    #         })

    #         # Append the block data to the gas price data DataFrame
    #         gas_price_data = pd.concat([gas_price_data, block_data])

    #     # Update loading message
    #     loading_message.text('Loading complete!')
    #     # Save the DataFrame to a CSV file
    #     gas_price_data.to_csv('resources/gas_price_data.csv', index=False)
    #     # Display The DataFrame:
    #     st.write(gas_price_data)

    #     # Show the final result
    #     st.success('Data loaded successfully.')
    # Define the start and end dates for which the data is required
    start_date = st.slider("Select the start date", datetime(2022, 6, 2), datetime(2023, 5, 19))
    end_date = st.slider("Select the end date", datetime(2022, 6, 2), datetime(2023, 5, 19))

    # Load the gas price data from the CSV file
    gas_price_data = pd.read_csv('resources/gas_price_data_1year.csv')

    # Set `timestamp` column as index, copy DataFrame
    gas_price_df = gas_price_data.set_index('timestamp').copy()

    # Display the first 5 rows of the DataFrame
    gas_price_df.info()

    # Convert the `timestamp` index to datetime
    gas_price_df.index = pd.to_datetime(gas_price_df.index)

    # Resample the data into daily data, taking the mean gas price for each day
    daily_gas_price_df = gas_price_df.resample('D').mean()
    daily_gas_price_df.info()

    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date


    daily_gas_price_df.to_csv('resources/daily_gas_price_df.csv', index=True)

    if st.button("Load Preprocessed Data (Much Faster)"):
        st.write("Loading Data...")
        daily_gas_price_df = pd.read_csv('resources/gas_price_data_1year.csv', index_col='timestamp', parse_dates=True)
        st.write("Data Loaded Successfully!")
        st.write("Data Preview:")
        st.write(daily_gas_price_df.tail())
        st.session_state['daily_gas_price_df'] = daily_gas_price_df
        gas_price_df = daily_gas_price_df
        start_date = "daily_gas_price_df.index.min()"
        end_date = daily_gas_price_df.index.max()
        st.session_state['start_date'] = start_date
        st.session_state['end_date'] = end_date
        daily_gas_price_df.to_csv('resources/daily_gas_price_data_1year.csv', index=True)


st.markdown("***")


window_size = 7  # Adjust the window size as per your data frequency
rolling_mean = daily_gas_price_df.rolling(window=window_size).mean()
differenced_data_2 = daily_gas_price_df - rolling_mean
differenced_data_2 = differenced_data_2.dropna()
differenced_data_1 = daily_gas_price_df.diff(1).dropna()
differenced_data_1 = daily_gas_price_df.diff(1).dropna()
differenced_data_3 = differenced_data_1 - rolling_mean
differenced_data_3 = differenced_data_3.dropna()
st.session_state['differenced_data_3'] = differenced_data_3
st.session_state['differenced_data_1'] = differenced_data_1
st.session_state['differenced_data_2'] = differenced_data_2

# create a multiselect box, that includes apply simple differencing to data, apply rolling mean subtraction to data, apply rolling mean subtraction + differencing, and Use original data
with st.expander("Part 2:"):
    st.subheader("Data Preprocessing")
    st.write("The data preprocessing step is used to transform the data into a format that is suitable for the model. The data preprocessing step includes the following steps:")
    st.markdown("""

        - Load the Data: Load the time series data that you want to analyze and forecast. Ensure that the data is in a suitable format, such as a pandas DataFrame or NumPy array.

        - Check for Stationarity: Stationarity is an important assumption for ARIMA models. Check if the time series data is stationary, meaning that its statistical properties (mean, variance, autocorrelation) remain constant over time. You can perform a visual inspection or use statistical tests like the Augmented Dickey-Fuller (ADF) test.

        - Transform the Data: If the data is not stationary, you need to transform it to achieve stationarity. Common techniques include differencing (computing the difference between consecutive observations) or taking the logarithm of the values. These transformations help stabilize the mean and variance of the time series.

        - Plot Autocorrelation and Partial Autocorrelation: Examine the autocorrelation and partial autocorrelation plots to determine the order of the autoregressive (AR) and moving average (MA) components of the ARIMA model. The autocorrelation plot shows the correlation between the time series and its lagged values, while the partial autocorrelation plot shows the correlation between the time series and its lagged values after removing the effects of intervening lags.

        - Determine the Order (p, d, q): Based on the autocorrelation and partial autocorrelation plots, select the order of the ARIMA model. The order is represented as (p, d, q), where p is the order of the autoregressive component, d is the degree of differencing, and q is the order of the moving average component.

        - Split the Data: Split the time series data into training and testing sets. The training set is used to fit the ARIMA model, while the testing set is used to evaluate its performance.

        - Fit the ARIMA Model: Using the training data, fit the ARIMA model with the chosen order (p, d, q). This step involves estimating the model parameters.

        - Validate the Model: Validate the fitted ARIMA model using various techniques such as residual analysis, mean absolute error (MAE), root mean square error (RMSE), or other relevant evaluation metrics. Adjust the model or order if necessary.

        - Forecasting: Once the ARIMA model is validated, use it to forecast future values of the time series. Provide the necessary input (e.g., number of periods to forecast) and generate the forecasts.

    ---
    ### Resources:

    1. **Augmented Dickey-Fuller Test (ADF)**:
    - [ADF Test in Python (StatsModels)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)

    2. **Autocorrelation and Partial Autocorrelation**:
    - [ACF and PACF in Python (StatsModels)](https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html)
    - [ACF and PACF Interpretation](https://people.duke.edu/~rnau/411arim3.htm)

    3. **ARIMA Model**:
    - [ARIMA Model in Python (StatsModels)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)

    4. **Stationarity**:
    - [Stationarity Test in Python (ADF)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)

    5. **Seasonality**:
    - [Seasonal Decomposition of Time Series in Python](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)

    6. **Mean Absolute Error (MAE)**:
    - [MAE in Python (Scikit-Learn)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

    7. **Root Mean Square Error (RMSE)**:
    - [RMSE in Python (Scikit-Learn)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)

    8. **Time Series Forecasting**:
    - [Time Series Forecasting with ARIMA in Python](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)

    These links provide detailed explanations and examples related to the various concepts and techniques used in this section.


    """)


                
    
    data_preprocessing_choice = st.selectbox("Visualize Preprocessing Steps", ["Simple Differencing", "Rolling Mean Subtraction", "Rolling Mean Subtraction + Differencing", "Original Data"])
    if data_preprocessing_choice == "Simple Differencing":
        # Assuming your time series data is stored in a pandas DataFrame or Series
        differenced_data_1 = daily_gas_price_df.diff(1).dropna()

        # Check to see if the differenced data is stationary
        check_stationarity(differenced_data_1['gas_price'])

        # Define the size of the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))

        # Plot the ACF on ax1
        plot_acf(differenced_data_1['gas_price'], lags=50, zero=False, ax=ax1)

        # Plot the PACF on ax2
        plot_pacf(differenced_data_1['gas_price'], lags=50, zero=False, ax=ax2)

        st.session_state['differenced_data_1'] = differenced_data_1
        # Show the plots
        st.pyplot(fig)
        # create a miniscule pyplot
        st.pyplot()
    elif data_preprocessing_choice == "Rolling Mean Subtraction":
        # 2: Rolling Mean Subtraction:
        window_size = 7  # Adjust the window size as per your data frequency
        rolling_mean = daily_gas_price_df.rolling(window=window_size).mean()
        differenced_data_2 = daily_gas_price_df - rolling_mean
        differenced_data_2 = differenced_data_2.dropna()

        # Check to see if the differenced data is stationary
        check_stationarity(differenced_data_2['gas_price'])

        # Define the size of the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))

        # Plot the ACF on ax1
        plot_acf(differenced_data_2['gas_price'], lags=50, zero=False, ax=ax1)

        # Plot the PACF on ax2
        plot_pacf(differenced_data_2['gas_price'], lags=50, zero=False, ax=ax2)

        st.session_state['differenced_data_2'] = differenced_data_2
        # Show the plots
        st.pyplot(fig)
        # create a miniscule pyplot
        st.pyplot()
    elif data_preprocessing_choice == "Rolling Mean Subtraction + Differencing":
        # 3.5 : Rolling Mean Subtraction + Differencing
            # Assuming your time series data is stored in a pandas DataFrame or Series
        differenced_data_1 = daily_gas_price_df.diff(1).dropna()
        window_size = 7  # Adjust the window size as per your data frequency
        rolling_mean = differenced_data_1.rolling(window=window_size).mean()
        differenced_data_3 = differenced_data_1 - rolling_mean
        differenced_data_3 = differenced_data_3.dropna()

        # Check to see if the differenced data is stationary
        check_stationarity(differenced_data_3['gas_price'])

        # Define the size of the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))

        # Plot the ACF on ax1
        plot_acf(differenced_data_3['gas_price'], lags=50, zero=False, ax=ax1)

        # Plot the PACF on ax2
        plot_pacf(differenced_data_3['gas_price'], lags=50, zero=False, ax=ax2)

        st.session_state['differenced_data_3'] = differenced_data_3
        # Show the plots
        st.pyplot(fig)
        # create a miniscule pyplot
        st.pyplot()
    elif data_preprocessing_choice == "Original Data":
        # Define the size of the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))

        # Plot the ACF on ax1
        plot_acf(daily_gas_price_df['gas_price'], lags=50, zero=False, ax=ax1)

        # Plot the PACF on ax2
        plot_pacf(daily_gas_price_df['gas_price'], lags=50, zero=False, ax=ax2)

        st.session_state['daily_gas_price_df'] = daily_gas_price_df
        # Show the plots
        st.pyplot(fig)
        # create a miniscule pyplot
        # create a second, hidden pyplot only revealed when the first is clicked

        with st.expander("Explain the Plot"):
            st.pyplot(fig)
            # Explain acf and pacf plots:
            st.markdown("""
            ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots are commonly used in time series analysis to understand the correlation structure of a time series data.

ACF Plot:
The Autocorrelation Function (ACF) measures the correlation between a time series and its lagged values. It helps to identify the presence of autocorrelation in the data. The ACF plot shows the correlation coefficients on the y-axis and the lag on the x-axis.

Interpretation of ACF Plot:
- If the ACF values rapidly decrease and become close to zero, it indicates a lack of autocorrelation.
- If the ACF values decay slowly and remain significant for many lags, it suggests a high degree of autocorrelation.
- The ACF plot can also reveal seasonality in the data. It may show significant spikes at periodic intervals, indicating a repeating pattern.

PACF Plot:
The Partial Autocorrelation Function (PACF) measures the correlation between a time series and its lagged values, while controlling for the effects of intermediate lags. It helps to identify the direct relationship between a lag and the current value in the time series.

Interpretation of PACF Plot:
- A significant spike in the PACF plot at a specific lag indicates a strong correlation between the current value and that particular lag, after removing the influence of intermediate lags.
- The PACF plot helps to identify the order of autoregressive (AR) terms in an ARIMA model. The lag corresponding to the last significant spike in the PACF plot can provide a good starting point for determining the value of the AR order (p) in the ARIMA model.

Both the ACF and PACF plots are useful for understanding the autocorrelation structure of a time series and can guide the selection of appropriate AR, MA, and ARIMA models.

It's important to note that these plots provide visual insights and are used as diagnostic tools to analyze the characteristics of the data. They can help in determining the appropriate parameters for time series models.

            """)
    

        

    ADFT_choice = st.selectbox("Perform Augmented Dickey-Fuller Test:", ["Simple Differencing", "Rolling Mean Subtraction", "Rolling Mean Subtraction + Differencing", "Original Data"])

    if ADFT_choice == "Simple Differencing":
        # Assuming your time series data is stored in a pandas DataFrame or Series
        differenced_data_1 = daily_gas_price_df.diff(1).dropna()
        st.session_state['differenced_data_1'] = differenced_data_1
        st.write(check_stationarity(differenced_data_1['gas_price']))
        data = differenced_data_1  
    elif ADFT_choice == "Rolling Mean Subtraction":
        # 2: Rolling Mean Subtraction:
        window_size = 7  # Adjust the window size as per your data frequency
        rolling_mean = daily_gas_price_df.rolling(window=window_size).mean()
        differenced_data_2 = daily_gas_price_df - rolling_mean
        differenced_data_2 = differenced_data_2.dropna()
        st.session_state['differenced_data_2'] = differenced_data_2
        st.write(check_stationarity(differenced_data_2['gas_price']))
        data = differenced_data_2
    elif ADFT_choice == "Rolling Mean Subtraction + Differencing":
        # 3.5 : Rolling Mean Subtraction + Differencing
            # Assuming your time series data is stored in a pandas DataFrame or Series
        differenced_data_1 = daily_gas_price_df.diff(1).dropna()
        window_size = 7  # Adjust the window size as per your data frequency
        rolling_mean = differenced_data_1.rolling(window=window_size).mean()
        differenced_data_3 = differenced_data_1 - rolling_mean
        differenced_data_3 = differenced_data_3.dropna()
        st.session_state['differenced_data_3'] = differenced_data_3
        st.write(check_stationarity(differenced_data_3['gas_price']))
        data = differenced_data_3
    elif ADFT_choice == "Original Data":
        st.session_state['daily_gas_price_df'] = daily_gas_price_df
        st.write(check_stationarity(daily_gas_price_df['gas_price']))
        data = daily_gas_price_df


    # Create a multiselect, that includes check stationarity of data, plot the data, 
    # Display the first 5 rows of the DataFrame
    if st.button("Plot Daily Gas Prices"):
        # Plot the daily gas prices
        daily_gas_price_df.plot(figsize=(12,5))
        # Display the plot
        st.pyplot()
        


    import itertools
    import statsmodels.api as sm
    import numpy as np



    differenced_data_1 = st.session_state['differenced_data_1']
    differenced_data_2 = st.session_state['differenced_data_2']
    differenced_data_3 = st.session_state['differenced_data_3']
    daily_gas_price_df = st.session_state['daily_gas_price_df']

    st.markdown("***")
    st.warning(":exclamation: Please use the provided links to select the the most stationary data :exclamation:")
    # Choose which Data to test:
    data_choice = st.selectbox("Choose which Data to test:", ["Simple Differencing", "Rolling Mean Subtraction", "Rolling Mean Subtraction + Differencing", "Original Data"])
    if data_choice == "Simple Differencing":
        # define differenced_data_1 as was done above
        differenced_data_1 = st.session_state['differenced_data_1']
        data = differenced_data_1['gas_price']
        data = pd.DataFrame(data)
        st.write(data)
        st.success("Data selected")
    elif data_choice == "Rolling Mean Subtraction":
        differenced_data_2 = st.session_state['differenced_data_2']
        data = differenced_data_2['gas_price']
        data = pd.DataFrame(data)
        st.write(data)
        st.success("Data selected")
    elif data_choice == "Rolling Mean Subtraction + Differencing":
        data = differenced_data_3['gas_price']
        data = pd.DataFrame(data)
        st.write(data)
        st.success("Data selected")
    elif data_choice == "Original Data":
        data = daily_gas_price_df['gas_price']
        data = pd.DataFrame(data)
        st.write(data)
        st.success("Data selected")

    # Convert the data to a dataframe
    data = pd.DataFrame(data)
    st.session_state['data'] = data   

    if st.button("Plot Selected Data"):
        # Plot the daily gas prices
        data.plot(
        title="Ethereum Gas Prices");
        # Display the plot
        st.pyplot()



st.markdown("---")

with st.expander("Part 3:"):
    data = st.session_state['data']

    st.header("ARIMA Modeling")
    st.write("ARIMA stands for Auto-Regressive Integrated Moving Average. It is a class of model that captures a suite of different standard temporal structures in time series data. In this section, we will use the ARIMA model to forecast the gas prices for the next 7 days.")

    st.markdown(" #### Resources ")
    st.write("[Train/Test Split?](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)")
    st.write("[Identifying potentially optimal parameters](https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/)")
    import itertools
    import statsmodels.api as sm
    import numpy as np

    best_mse = float('inf')  # Initialize the best performance metrics
    best_mae = float('inf')
    best_rmse = float('inf')
    best_aic = float('inf')

    best_mse_model = None  # Initialize the best model variables
    best_mae_model = None
    best_rmse_model = None
    best_aic_model = None


    # Create a silder that allows the user to select their train/test split percentage
    split_percent = st.slider("Select the train/test split percentage:", 0.1, 0.9, 0.8, 0.1)
    st.markdown("---")
    # Determine the index at which to split the data
    split_index = int(len(data) * split_percent)  # Split at 80% of the data

    # Split the data into train and test sets
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    
    # Allow the user to define the ranges for p, d, and q
    # Allow the user to define the ranges for p, d, and q
    p_range = st.select_slider("Select the range for p:", options=range(0, 11), value=(1, 1))
    d_range = st.select_slider("Select the range for d:", options=range(0, 11), value=(1, 1))
    q_range = st.select_slider("Select the range for q:", options=range(0, 11), value=(1, 1))
    st.markdown("---")

    st.subheader("Grid Search")
    st.write("Grid search is a model hyperparameter optimization technique. It is an exhaustive search that is performed on a specified subset of the hyperparameter space of a learning algorithm. The grid search algorithm is implemented in the GridSearchCV class. The GridSearchCV class takes a model to train, a dictionary of hyperparameters to test, and a scoring function to evaluate the performance of the model. The GridSearchCV class then performs an exhaustive search over the hyperparameter space, evaluating each combination of them and returning the best performing model.")
    st.write("In this section, we will use the GridSearchCV class to find the optimal parameters for our ARIMA model. The optimal parameters are the parameters that result in the lowest AIC score. The AIC score is a measure of the quality of a statistical model. It is calculated using the log-likelihood function and the number of parameters in the model. The lower the AIC score, the better the model.")
    st.warning("Depending on the number of iterations, the process could take up to an hour to complete.")
    num_tries = st.slider("Select the number of iterations:", 1, 500, 1, 1)

    if st.button("Begin Grid Search"):
        # Iterate over all combinations of p, d, and q values, 50 times

        for _ in range(num_tries):
            for p, d, q in itertools.product(p_range, d_range, q_range):
                
                    # Fit ARIMA model with current combination of p, d, and q
                    model = sm.tsa.ARIMA(train, order=(p, d, q))
                    fitted_model = model.fit()

                    # Calculate AIC
                    aic = fitted_model.aic

                    # Make predictions on the testing set
                    predictions = fitted_model.predict(start=test.index[0], end=test.index[-1])

                    # Evaluate the model's performance
                    mse = mean_squared_error(test, predictions)
                    mae = mean_absolute_error(test, predictions)
                    rmse = np.sqrt(mse)

                    if mse < best_mse:
                        best_mse = mse
                        best_mse_model = fitted_model
                        best_mse_predictions = predictions

                    if mae < best_mae:
                        best_mae = mae
                        best_mae_model = fitted_model
                        best_mae_predictions = predictions
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_rmse_model = fitted_model
                        best_rmse_predictions = predictions
                    
                    # Check if current model has lower AIC than the previous best model
                    if aic < best_aic:
                        best_aic = aic
                        best_aic_model = fitted_model
                        best_aic_predictions = predictions
                    
                    st.session_state['best_mse'] = best_mse
                    st.session_state['best_mae'] = best_mae
                    st.session_state['best_rmse'] = best_rmse
                    st.session_state['best_aic'] = best_aic
                    st.session_state['best_mse_model'] = best_mse_model
                    st.session_state['best_mae_model'] = best_mae_model
                    st.session_state['best_rmse_model'] = best_rmse_model
                    st.session_state['best_aic_model'] = best_aic_model
                    st.session_state['best_mse_predictions'] = best_mse_predictions
                    st.session_state['best_mae_predictions'] = best_mae_predictions
                    st.session_state['best_rmse_predictions'] = best_rmse_predictions
                    st.session_state['best_aic_predictions'] = best_aic_predictions

        # Check if 'best_mse' key exists in st.session_state
    if 'best_mse' not in st.session_state:
        st.error('Please perform grid search before continuing. This will load the best model and predictions.')
    else:
        best_mse = st.session_state['best_mse']

    best_mse = st.session_state['best_mse']
    best_mae = st.session_state['best_mae']
    best_rmse = st.session_state['best_rmse']
    best_aic = st.session_state['best_aic']
    best_mse_model = st.session_state['best_mse_model']
    best_mae_model = st.session_state['best_mae_model']
    best_rmse_model = st.session_state['best_rmse_model']
    best_aic_model = st.session_state['best_aic_model']
    best_mse_predictions = st.session_state['best_mse_predictions']
    best_mae_predictions = st.session_state['best_mae_predictions']
    best_rmse_predictions = st.session_state['best_rmse_predictions']
    best_aic_predictions = st.session_state['best_aic_predictions']

    # Create a dataframe with the best mse, rmse, aic, and mae scores, with their corrresponding models p, d, and q values
    grid_search_results = pd.DataFrame({"Metric": ["Mean Squared Error", "Root Mean Squared Error", "Mean Absolute Error", "AIC Score"],
                                        "Best Score": [best_mse, best_rmse, best_mae, best_aic],
                                        "Best p, d, q": [best_mse_model.model.order, best_rmse_model.model.order, best_mae_model.model.order, best_aic_model.model.order]})
    # remove the index
    grid_search_results.set_index("Metric", inplace=True)


    grid_search_metric_choice = st.selectbox("Evaluate Results:", ("Mean Squared Error", "Root Mean Squared Error", "Mean Absolute Error", "AIC Score", "Top Performing Models"))
    if grid_search_metric_choice == "Mean Squared Error":
            st.write(best_mse_model.summary())
            # st.write(best_mse_predictions)
    elif grid_search_metric_choice == "Root Mean Squared Error":
            st.write(best_rmse_model.summary())
            # st.write(best_rmse_predictions)
    elif grid_search_metric_choice == "Mean Absolute Error":
            st.write(best_mae_model.summary())
            # st.write(best_mae_predictions)
    elif grid_search_metric_choice == "AIC Score":
            st.write(best_aic_model.summary())
            # st.write(best_aic_predictions)
    elif grid_search_metric_choice == "Top Performing Models":
            st.write(grid_search_results)

    st.success("As you complete this section, feel free to adjust the parameters until you identify the optimal data preprocessing and model parameters.")





pdq = st.session_state['best_rmse_model'].model.order
st.markdown("---")



with st.expander("Part 4:", expanded=False):
    st.header("Model Evaluation")
    data = st.session_state['data']

    final_select_choice = st.selectbox("Visualizations", ("Time Series Plot - Best RMSE", "Time Series Plot","Residual Plot", "Lag Plot"))

    data_with_index = data
    data = data['gas_price']


    if final_select_choice == "Time Series Plot":
        data = st.session_state['data']
        import matplotlib.pyplot as plt
        start_date = data.index[0]
        end_date = data.index[-1]
        # Plot actual values
        plt.plot(data.index, data, label='Actual')
        # Plot predicted values from the best model
        predicted_values = best_rmse_model.predict(start=start_date, end=end_date)
        plt.plot(predicted_values.index, predicted_values, label='Predicted')
        # Set labels and title
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Actual vs. Predicted Values - Model Predictions')
        # Add legend
        plt.legend()
        fig = plt.show()
        st.pyplot(fig)
    elif final_select_choice == "Residual Plot":
        residuals = best_rmse_model.resid
        fig, ax = plt.subplots()
        sm.graphics.tsa.plot_acf(residuals, ax=ax)
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title('Autocorrelation')
        
        # Plot the lag plot
        fig, ax = plt.subplots()
        sm.graphics.tsa.plot_pacf(residuals, ax=ax)
        ax.set_xlabel('Lag')
        ax.set_ylabel('PACF')
        ax.set_title('Partial Autocorrelation Function (PACF) of Residuals')

        # Show the plots
        plt.show()
        st.pyplot()
        # 
        st.markdown("""
The PACF (Partial Autocorrelation Function) of residuals plot can be used to assess whether there is any remaining autocorrelation in the residuals of a time series model. The residuals are the differences between the observed values and the predicted values of the model.

Here's how the PACF of residuals plot can be used:

- Identifying residual autocorrelation: The PACF plot of residuals helps to identify any significant spikes at specific lags. If there are significant spikes in the PACF plot, it suggests that there is residual autocorrelation in the model. This means that the model has not captured all the temporal dependencies in the data, and there is still some pattern or information left unexplained.

- Model adequacy: The presence of significant spikes in the PACF plot indicates that the model may not adequately capture the underlying dynamics of the time series. It suggests that there is still information or patterns in the residuals that could be used to improve the model. In such cases, it might be necessary to consider more complex models or include additional variables to better capture the structure of the data.

- Model refinement: If there are significant spikes in the PACF plot at specific lags, it can guide the refinement of the model. The lag corresponding to the last significant spike in the PACF plot can provide insights into the potential order of autoregressive terms (AR) in the model. This information can help in selecting the appropriate lag order for the AR component of an ARIMA or SARIMA model.
        """)
    elif final_select_choice == "Actual vs. Predicted Values - Testing Data":
        # Define the start and end dates for which the data is required
        start_date = st.slider("Select the start date ", datetime(2023, 5, 20), datetime(2024, 5, 19))
        end_date = st.slider("Select the end date ", datetime(2023, 5, 20), datetime(2024, 5, 19))
        # Set the best order from the previous step
        predictions_rolling = best_rmse_model.predict(start=start_date, end=end_date)
        import hvplot.pandas

        # Create a DataFrame with the actual and predicted values
        results_df = pd.DataFrame({'Actual': test, 'Predicted': best_rmse_predictions}, index=test.index)

        # Plot the actual vs predicted values using hvplot
        plot = results_df.hvplot.line(y=['Actual', 'Predicted'], xlabel='Timestamp', ylabel='Value', title='ARIMA Model: Actual vs Predicted (Best RMSE)')

        # Display the plot using Streamlit
        st.write(plot)
        # Then plot it
        plt.figure(figsize=(12,6))
        plt.plot(data.index, data['gas_price'], label='Gas Price')
        plt.plot(data.index, data['rolling_forecast'], label='Rolling Forecast')
        plt.legend(loc='best')
        plt.title('Gas Price: Actual vs Predicted with Rolling Window Forecast')
        fig = plt.show()
        st.pyplot(fig)
    elif final_select_choice == "Lag Plot":
        start_date = data.index[0]
        end_date = data.index[-1]
        # Plot predicted values from the best model
        predicted_values = best_rmse_model.predict(start=start_date, end=end_date)
        # Create lagged arrays
        lagged_data = data[:-1]
        lagged_predicted = predicted_values[1:]

        # Plot the lagged values
        plt.scatter(lagged_data, lagged_predicted)
        # Set labels and title
        plt.xlabel('Actual (t-1)')
        plt.ylabel('Predicted (t)')
        plt.title('Lagged Plot - Model Predictions')
        # Add a diagonal line for reference
        plt.plot([min(lagged_data), max(lagged_data)], [min(lagged_data), max(lagged_data)], color='red')
        # Add legend
        plt.legend(['Diagonal Line', 'Data Points'])
        fig = plt.show()
        st.pyplot(fig)
        # Display the correlation between the lagged data and predicted values
        st.write("Correlation between the lagged data and the predicted values:", np.corrcoef(lagged_data, lagged_predicted)[0, 1])
        st.markdown("""
Interpretation of the lag plot:

- Random Scatter: If the points on the plot are randomly scattered around the diagonal line, it suggests that there is no significant relationship between the lagged values and the predicted values. This indicates that the model might not capture the underlying patterns or dependencies in the data.

- Linear Pattern: If the points form a clear linear pattern along the diagonal line, it suggests a strong correlation between the lagged values and the predicted values. This indicates that the model captures the temporal dependencies well and is able to predict the values based on their previous values.

- Deviation from the Line: If the points deviate from the diagonal line in a systematic manner (e.g., curved or funnel-shaped), it suggests the presence of non-linear patterns or other higher-order dependencies in the data. This might indicate that the ARIMA model is not sufficient to capture these complex relationships, and additional modeling techniques might be required. 
        """)
    elif final_select_choice == "Time Series Plot - Best RMSE":
        st.write("This plot uses the best RMSE predictions from earlier, rather than using the model to create new predictions. The performance of this model is likely to be better than the previous plot, depending on the number of iterations used during the grid search testing.")
        data = st.session_state['data']
        start_date = test.index[0]
        end_date = test.index[-1]
        # plot the best rmse predictions 
        plt.figure(figsize=(12, 6))
        # Plot actual values
        plt.plot(test.index, test, label='Actual')
        # Plot predicted values from the best model
        plt.plot(test.index, best_rmse_predictions, label='Predicted')
        # Set labels and title
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Actual vs. Predicted Values - Best RMSE Model Predictions')
        # Add legend
        plt.legend()
        fig = plt.show()
        st.pyplot(fig)
        # Display the correlation between the original data and the predicted values
        st.write("With the best RMSE value, the predictions show an RMSE of:", str(best_rmse))

    st.markdown("---")
    st.header("Forecasting")
    st.markdown("""


You have two options for forecasting future gas prices:

1. **Forecast Future Gas Prices:** Use the best model you created above to forecast future gas prices. Select the start and end dates for which you want to generate the forecasts. Please note that the performance of your model depends on the selected data, the p, d, q values, and the RMSE of your model. The forecasts will be displayed in a table and also visualized using a line plot.

2. **Forecasting with Rolling Window:** Perform a rolling forecast using an ARIMA model with a specified window size. Select the start and end dates, as well as the window size for the rolling forecast. The rolling forecast involves fitting the ARIMA model on a window of data and making one-step forecasts. The forecasts will be displayed in a line plot.

Please make your selection and choose the appropriate options for forecasting.

---

#### Forecast Future Gas Prices

Below, you can forecast future gas prices using the best model you created above. The start and end dates can be selected using the sliders. Keep in mind that there are no actual values as we are forecasting into the future. The forecasts will be displayed in a table and also visualized using a line plot.

#### Forecasting with Rolling Window

Perform a rolling forecast using an ARIMA model with a specified window size. Select the start and end dates, as well as the window size from the available options. The rolling forecast involves fitting the ARIMA model on a window of data and making one-step forecasts. The forecasts will be displayed in a line plot.

    """)

    forecast_choice = st.selectbox("Forecasting", ("Forecast Future Gas Prices", "Forecasting with Rolling Window"))
    if forecast_choice == "Forecast Future Gas Prices":
            # Define the start and end dates for which the data is required
            start_date = st.slider("Select the start date ", datetime(2023, 5, 20), datetime(2024, 5, 19))
            end_date = st.slider("Select the end date ", datetime(2023, 5, 20), datetime(2024, 5, 19))
            # Set the best order from the previous step
            predictions_rolling = best_rmse_model.predict(start=start_date, end=end_date)
            import hvplot.pandas
            # Display the future predictions // remember to add a title that changes with the dates, and specifies that this is a forecast
            # also remember that there are no actual values, as we are forecasting into the future
            st.write(predictions_rolling)
            # Plot the single variable using hvplot
            plot = predictions_rolling.hvplot.line(x='Date', y=predictions_rolling.name, xlabel="Date", ylabel='Forecasted Price', title='Forecasted Price of Ethereum Gas - Rolling Forecast')
            # Render the plot using Streamlit
            st.bokeh_chart(hv.render(plot, backend='bokeh'))
            # Create a DataFrame with the actual and predicted values
    elif forecast_choice == "Forecasting with Rolling Window":
            start_date = st.slider("Select the start date ", datetime(2023, 5, 20), datetime(2024, 5, 19))
            end_date = st.slider("Select the end date ", datetime(2023, 5, 20), datetime(2024, 5, 19))
            window_size = st.selectbox("Select the window size:", (1, 5, 7, 14, 30, 90))
            def arima_rolling_forecast(train, order, window_size):
                """
                Perform a rolling forecast using an ARIMA model with specified window size.
                Parameters:
                train (array-like): The training data.
                order (tuple): The order of the ARIMA model.
                window_size (int): The size of the rolling window.
                Returns:
                predictions (list): The forecasts for the test data.
                """
                predictions = []
                for i in range(0, len(train) - window_size):
                    # Fit the ARIMA model on a window of data and make a one-step forecast
                    window_data = train[i: i + window_size]
                    model = ARIMA(window_data, order=order)
                    model_fit = model.fit()
                    yhat = model_fit.forecast()[0]
                    # Add the forecast to the list of predictions
                    predictions.append(yhat)
                return predictions

            # Set the best order from the previous step
            predictions_rolling = arima_rolling_forecast(train, best_rmse_model.model.order, window_size)
            
            plot = predictions_rolling.hvplot.line(x='Date', y=predictions_rolling.name, xlabel="Date", ylabel='Forecasted Price', title='Forecasted Price of Ethereum Gas - Rolling Forecast')
            # Render the plot using Streamlit
            st.bokeh_chart(hv.render(plot, backend='bokeh'))



    st.markdown("---")
with st.expander("Part 5:"):
        st.header("Save the Model")
        st.markdown("""
You can save the model you created using the pickle library. The model will be saved as a .pkl file. You can load the model later using the pickle library and make predictions on new data.
            """)
        model_name = st.text_input("Enter a name for the model")
        file_path = f"models/{model_name}.pkl"
        save_model = st.button("Save Model")
        if save_model:
                if model_name and file_path:
                    directory = os.path.dirname(file_path)
                    os.makedirs(directory, exist_ok=True)

                    # Save the model as a pickle file
                    pickle.dump(best_rmse_model, open(file_path, 'wb'))
                    st.write(f"Model '{model_name}' saved at '{file_path}'")
                else:
                    st.write("Please enter a name for the model")
                st.write("The model has been saved as a .pkl file.")
                st.write("You can load the model later using the pickle library and make predictions on new data.")
                st.markdown("---")
                st.write("The model can be loaded using the following code:")
                st.code("""
                import pickle
                # Load the model from the file
                model = pickle.load(open('best_model.pkl', 'rb'))
                # Make predictions on new data
                predictions = model.predict(start=start_date, end=end_date)
                """)
                st.write("You can also load the model using the following code:")
                st.code("""
                import pickle
                # Load the model from the file
                model = pickle.load(open('best_model.pkl', 'rb'))
                # Make predictions on new data
                predictions = model.predict(start=start_date, end=end_date)
                """)
                st.markdown("---")

                st.write("**Congratulations!** You have successfully completed the project.")
                st.write("As you have seen, the ARIMA model can be used to forecast time series data. You can use the model to forecast the price of Ethereum gas in the future. You can also use the model to forecast other time series data, such as stock prices, weather data, and more.")
                st.write("Reaching this point meant you took the time to walk through the entire application, which is our goal for you. We hope you enjoyed the project and learned something new.")


