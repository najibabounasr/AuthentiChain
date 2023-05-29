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

    # Define the start and end dates for which the data is required
    start_date = st.slider("Select the start date", datetime(2020, 5, 19), datetime(2021, 5, 19))
    end_date = st.slider("Select the end date", datetime(2020, 5, 19), datetime(2021, 5, 19))
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date


    if st.button("Load live Prices Data") and (not pd.isnull(end_date)) and (not pd.isnull(start_date)):
        # Load .env file
        load_dotenv()

        # Load API key from .env file
        api_key = os.getenv("ALCHEMY_API_KEY")

        # Define the Alchemy API endpoint
        api_endpoint = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"

        # Set up Web3 connection
        w3 = Web3(Web3.HTTPProvider(api_endpoint))

        # Check if the connection is successful
        if w3.is_connected():
            st.write("Connected to Ethereum node.")
        else:
            st.write("Connection failed!")
        
        # Display loading message
        loading_message = st.text('Loading...')
        
        # Simulate some time-consuming process
        time.sleep(5)

        # Define the block time(in seconds)
        avg_block_time = 13  
        st.write(f"Average block time: {avg_block_time} seconds")

        # Calculate the estimated number of blocks per hourn(3600 seconds in an hour)
        blocks_per_hour = int(3600 / avg_block_time)
        st.write(f"Estimated blocks per hour: {blocks_per_hour}")

        # Calculate the total hours for the given time period
        total_hours = int((end_date - start_date).total_seconds() / 3600)

        # Calculate the total number of blocks for the given time period
        total_blocks = total_hours * blocks_per_hour
        st.write(f"Total number of hourly blocks for the given time period: {total_blocks}")

        # Get the latest block number
        latest_block = w3.eth.block_number
        st.write(latest_block)

        # Calculate the start block
        start_block = latest_block - total_blocks
        start_block

        # Initialize an empty DataFrame to store the gas prices
        gas_price_data = pd.DataFrame(columns=['timestamp', 'gas_price'])

        # Loop through the blocks, show status bar
        for block_number in tqdm(range(start_block, latest_block, blocks_per_hour),
                                desc="Fetching gas prices data"): 
            
            # Get the block details
            block = w3.eth.get_block(block_number)

            # Convert the gas price from hex to integer
            gas_price = int(str(block['baseFeePerGas']), 16)

            # Convert timestamp to datetime and create a DataFrame with the block timestamp and gas price
            block_data = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(block['timestamp'])],
                'gas_price': [gas_price]
            })

            # Append the block data to the gas price data DataFrame
            gas_price_data = pd.concat([gas_price_data, block_data])

        # Update loading message
        loading_message.text('Loading complete!')
        # Save the DataFrame to a CSV file
        gas_price_data.to_csv('resources/gas_price_data.csv', index=False)
        # Display The DataFrame:
        st.write(gas_price_data)

        # Show the final result
        st.success('Data loaded successfully.')

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


    daily_gas_price_df.to_csv('resources/daily_gas_price_df.csv', index=True)

st.markdown("***")
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
                
    
    data_preprocessing_choice = st.selectbox("Visualize Preprocessing Steps", ["Simple Differencing", "Rolling Mean Subtraction", "Apply Rolling Mean Subtraction + Differencing", "Original Data"])
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

    ADFT_choice = st.selectbox("Perform Augmented Dickey-Fuller Test:", ["Simple Differencing", "Rolling Mean Subtraction", "Rolling Mean Subtraction + Differencing", "Original Data"])

    if ADFT_choice == "Simple Differencing":
        # Assuming your time series data is stored in a pandas DataFrame or Series
        differenced_data_1 = daily_gas_price_df.diff(1).dropna()
        st.session_state['differenced_data_1'] = differenced_data_1
        st.write(check_stationarity(differenced_data_1['gas_price']))
    elif ADFT_choice == "Rolling Mean Subtraction":
        # 2: Rolling Mean Subtraction:
        window_size = 7  # Adjust the window size as per your data frequency
        rolling_mean = daily_gas_price_df.rolling(window=window_size).mean()
        differenced_data_2 = daily_gas_price_df - rolling_mean
        differenced_data_2 = differenced_data_2.dropna()
        st.session_state['differenced_data_2'] = differenced_data_2
        st.write(check_stationarity(differenced_data_2['gas_price']))
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
    elif ADFT_choice == "Original Data":
        st.session_state['daily_gas_price_df'] = daily_gas_price_df
        st.write(check_stationarity(daily_gas_price_df['gas_price']))


    # Create a multiselect, that includes check stationarity of data, plot the data, 
    # Display the first 5 rows of the DataFrame
    if st.button("Plot Daily Gas Prices"):
        # Plot the daily gas prices
        gas_price_df['gas_price'].plot(
        title="Etherium hourly gas prices for 1 yaer period");
        adf_test = adfuller(daily_gas_price_df['gas_price'])
        # Display the plot
        st.pyplot()
        st.write(f'p-value: {adf_test[1]}')


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
    num_tries = st.slider("Select the number of iterations:", 1, 30, 1, 1)

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

    
with st.expander("Model Evaluation", expanded=False):

    final_select_choice = st.selectbox("Visualizations", ("Actual vs. Predicted Values", "Actual vs. Predicted Values - Rolling Forecast", "Mean Absolute Error"))

    if final_select_choice == "Actual vs. Predicted Values":
            import matplotlib.pyplot as plt
            start_date = st.session_state['start_date']
            end_date = st.session_state['end_date']
        
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

            # Display the plot in Streamlit
            st.pyplot()
            