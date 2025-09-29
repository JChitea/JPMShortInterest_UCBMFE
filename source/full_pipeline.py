##
# Relevant Libraries
##
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL as pl
import base64
import json
import anthropic
from typing import Generator, Optional
import requests
from requests.auth import HTTPBasicAuth
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import io
##

# Load in the .env file
from dotenv import load_dotenv
load_dotenv()
# Load in the API keys
global ALPHA_API, CLAUDE_API, alpha_base_url, finra_base_url
global FINRA_CLIENT_ID, FINRA_CLIENT_PASS, claude_model, claude_max_tokens
ALPHA_API = os.getenv('ALPHA_API')
CLAUDE_API = os.getenv('CLAUDE_API')
FINRA_CLIENT_ID = os.getenv('FINRA_CLIENT_ID')
FINRA_CLIENT_PASS = os.getenv('FINRA_CLIENT_PASS')
claude_model = 'claude-sonnet-4-20250514'
claude_max_tokens = 52000


alpha_base_url = 'https://www.alphavantage.co/query'
finra_base_url = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"


##
# Functions relating to data aggregationg and processing
##

# We'll have a function that uses already downloaded short_interest data from FINRA for testing
def load_short_interest_data(symbol):
    # the data is store in the short_data directory with files named symbol_SI.csv
    file_path = f"short_data/{symbol}_SI.csv"
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['Settlement Date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    # We only need to save the settlement date and the short interest
    df = df[['Current Short']]
    df.rename(columns={'Current Short': 'short_interest'}, inplace=True)
    # I also want to make sure the short interest is a float
    df['short_interest'] = df['short_interest'].astype(float)   
    # And we need to save the description of the dataframe
    description = {
        "symbol": symbol,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).tolist(),
        "shape": df.shape
    }
    with open(f"in_tabular/{symbol}_description.json", "w") as f:
        json.dump(description, f)
    # and we need to plot the short interest
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['short_interest'])
    plt.title(f"{symbol} - Short Interest")
    plt.xlabel("Date")
    plt.ylabel("Short Interest")
    plt.grid()
    plt.savefig(f"in_images/{symbol}_short_interest.png")
    plt.close()
    return df

def get_FINRA_auth():

    # Step 1: Get OAuth2 token
    token_url = finra_base_url
    
    # OAuth2 client credentials flow
    auth = HTTPBasicAuth(FINRA_CLIENT_ID, FINRA_CLIENT_PASS)
    r = requests.post(url=token_url,
                      params={'grant_type': 'client_credentials'},
                      auth=auth)

    r.raise_for_status()
    return r.json()['access_token']

def get_FINRA_short_data(symbol, date_range=[None, None]):
    token = get_FINRA_auth()
    
    # Construct the payload
    base = 'https://api.finra.org/'
    endpoint = 'data/group/otcMarket/name/consolidatedShortInterest'
    url = base + endpoint
    
    # Build the request payload with filters
    payload = {
        'limit': 1000,
        'compareFilters': [
            {
                'compareType': 'EQUAL',
                'fieldName': 'symbolCode',
                'fieldValue': symbol.upper()
            }
        ]
    }
    
    # Add date filters if provided
    if date_range[0] is not None or date_range[1] is not None:
        date_filter = {'fieldName': 'settlementDate'}
        if date_range[0] and date_range[1]:
            date_filter['startDate'] = date_range[0]
            date_filter['endDate'] = date_range[1]
            payload['dateRangeFilters'] = [date_filter]
    
    # Submit the request
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'Accept': 'text/plain'
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    # Parse the CSV response
    df = pd.read_csv(
        io.StringIO(response.text),
        sep=",",
        engine="python",
        keep_default_na=False,
    )
    
    if not df.empty:
        # Return the relevant columns with renamed headers
        result_df = df[['settlementDate', 'symbolCode', 'currentShortPositionQuantity']].copy()
        result_df.columns = ['settlementDate', 'symbol', 'shortInterest']
        # Now process the data
        result_df = process_finra_data(result_df, symbol)   
        return result_df
    else:
        return pd.DataFrame()
    

def process_finra_data(df, symbol):
    if df.empty:
        raise ValueError("No data returned from FINRA API")
    df['date'] = pd.to_datetime(df['settlementDate'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    df = df[['shortInterest']]
    df.rename(columns={'shortInterest': 'short_interest'}, inplace=True)
    df['short_interest'] = df['short_interest'].astype(float)

    # Save description
    description = {
        "symbol": symbol,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).tolist(),
        "shape": df.shape
    }
    with open(f"in_tabular/{symbol}_description.json", "w") as f:
        json.dump(description, f)

    # Plot short interest
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['short_interest'])
    plt.title(f"{symbol} - Short Interest (FINRA)")
    plt.xlabel("Date")
    plt.ylabel("Short Interest")
    plt.grid()
    plt.savefig(f"in_images/{symbol}_short_interest.png")
    plt.close()

    return df   


def get_daily_stock_data(symbol, api_key=ALPHA_API, date_range=[None, None]):
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'full'
    }
    response = requests.get(alpha_base_url, params=params)
    data = response.json()
    if "Time Series (Daily)" not in data:
        raise ValueError("Error fetching data from Alpha Vantage API")
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    if date_range[0]:
        df = df[df.index >= pd.to_datetime(date_range[0])]
    if date_range[1]:
        df = df[df.index <= pd.to_datetime(date_range[1])]
    df = df.astype(float)
    df = df.sort_index()
    # Remove all the numbers and periods from the column names
    df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
    # Make sure the words in the column names are always seperated by underscores
    df.columns = [col.replace(' ', '_') for col in df.columns]
    return df

# Now, we have a way of getting data. The data that comes in ias going to need to be processed for the LLM. The requirements are:
# 1. We need a description of each column and the data frame (inlcuding dimensions and dtypes)
# 2. We need to visualize the data and make a plot for each column and save it as a PNG

def process_stock_data(df, symbol):
    description = {
        "symbol": symbol,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).tolist(),
        "shape": df.shape
    }
    with open(f"in_tabular/{symbol}_description.json", "w") as f:
        json.dump(description, f)

    for column in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[column])
        plt.title(f"{symbol} - {column}")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.grid()
        plt.savefig(f"in_images/{symbol}_{column}.png")
        plt.close()


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Now I we might need to clear all the images in the in_images directory at some point and the json files
def clear_images_and_json(symbols):
    for symbol in symbols:
        # Remove description file
        desc_file = f"in_tabular/{symbol}_description.json"
        if os.path.exists(desc_file):
            os.remove(desc_file)
        # Remove images
        for file in os.listdir("in_images"):
            if file.startswith(symbol) and file.endswith(".png"):
                os.remove(os.path.join("in_images", file))

##
# Functions relating to Claude API
##

def get_description_dict(symbol):
    # For every file in the in_tabular and in_images directory, we need to create a dictionary with keys as the file name and value as the description
    description_dict = {}
    for file in os.listdir("in_tabular"):
        # if we are in this loop, we are in the in_tabular directory
        if file == f"{symbol}_description.json":
            with open(os.path.join("in_tabular", file), "r") as f:
                description_json = f.read()
            type_obj = 'csv'
            data_obj = description_json
            description_dict['file'] = {
                'type': type_obj,
                'data': data_obj
            }
            break
    for file in os.listdir("in_images"):
        if file.startswith(symbol) and file.endswith(".png"):
            encoded_image = encode_image_to_base64(os.path.join("in_images", file))
            type_obj = 'image'
            data_obj = encoded_image
            description_dict[file] = {
                'type': type_obj,
                'data': data_obj
            }

    return description_dict

# We first need a function that is going to prep the prompt for the Claude API
def create_claude_payload(stock_list, short_stock, short_plot, max_number_of_features=10, last_error=""):
    base_prompt = f"""
    Generate features from the labeled stock data and images for a machine learning model to predict the 
    short interest of {short_stock}. The image of the short interest is labeled as {short_plot}. As an input 
    you will alawyas have a master data frame with the features you see in each data file prefixed by the stock symbol, so you need to observe
    what columns are in each data file and generate features based on those columns. Note that the short interest data is biweekly,
    while the stock data is daily, so you will need to aggregate the daily data to biweekly to match the short interest data. 
    So keep this in mind when generating features.  
    """
    response_fmt_prompt = f"""
    Please format your response as a python script ONLY, with each feature in its own method. 
    Nothing else should be included in the response except at the end a summary of your reasoning 
    from the provided data that made you choose these features. Generate at most {max_number_of_features} features.
    """
    full_prompt = base_prompt + response_fmt_prompt+ f"\n{last_error}"
    
    data_dict = {}

    # Now get the data dictionary
    for stock in stock_list:
        desc_dict = get_description_dict(stock)
        data_dict[stock] = desc_dict
    # Also add the short interest plot
    short_desc_dict = get_description_dict(short_stock)
    data_dict[short_stock] = short_desc_dict

    # Now create the payload
    content = []
    content.append({'type': 'text', 'text': full_prompt})

    # Fix: Iterate through the nested structure properly
    for stock, stock_data in data_dict.items():
        for file_key, file_info in stock_data.items():
            if file_info['type'] == 'csv':
                # Add text description for the CSV data
                content.append({
                    'type': 'text', 
                    'text': f"Data description for {stock}:\n{file_info['data']}"
                })
            elif file_info['type'] == 'image':
                # Use proper Anthropic Messages API format for images
                content.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': file_info['data']
                    }
                })
    
    claude_payload = {
        "model": claude_model,
        "max_tokens": claude_max_tokens,
        "messages": [
            {"role": "user", "content": content}
            # Test to see if system messages help better
            # {"role": "system", "content": content}
        ]
    }
    return claude_payload

def call_claude_api(payload):
    client = anthropic.Anthropic(api_key=CLAUDE_API)
    full_response = ""
    try:
        with client.messages.stream(**payload) as stream:
            for text in stream.text_stream:
                print(text, end='', flush=True)
                full_response += text
        return full_response
    except Exception as e:
        print(f"Error during Claude API call: {e}")
        return None
    
# We now need a function that is going to parse the claude response into a python script
def parse_response(response):

    start = response.find("```python")
    if start == -1:
        start = response.find("```")
        if start == -1:
            print("No code block found in the response.")
            return None
        else:
            end = response.find("```", start + 3)
            code_block = response[start + 10:end].strip()
    else:
        end = response.find("```", start + 9)
        code_block = response[start + 9:end].strip()
    
    # Now we need to save this code block as a python script
    with open("generated_features.py", "w") as f:
        f.write(code_block)
        print('Saved generated_features.py')
    
    return code_block

# We need a function that is going to clear the generated_features.py file
def clear_generated_features():
    if os.path.exists("generated_features.py"):
        os.remove("generated_features.py")  

## 
# Function relating to the pipeline post feature generation
##

# We need to recretae this function now that we know the structure of df_dict is a master dataframe with columns prefixed by the stock symbol
def create_engineered_features(df):
    import generated_features  
    feature_dfs = []
    # Put this into a try except block to catch any errors
    # Get a list of all functions in the generated_features module assuming nothing about the name of the functions
    feature_funcs = [getattr(generated_features, func) for func in dir(generated_features) if callable(getattr(generated_features, func))]
    for func in feature_funcs:
        feature_df = func(df)
        feature_dfs.append(feature_df)  

    # Concatenate all feature dataframes
    if feature_dfs:
        combined_df = pd.concat(feature_dfs, axis=1)
        return combined_df
    else:
        return pd.DataFrame()

# We need a function then that is going to use our engineered features to train an XGBOOST model to predict short interest
def train_xgboost_model(features_df, target_series):
    # Align features and target
    data = features_df.join(target_series, how='inner')
    
    # Add debugging
    print(f"\nAfter join - data shape: {data.shape}")
    if data.shape[0] == 0:
        print("ERROR: No matching dates between features and target!")
        print("Make sure your feature engineering functions aggregate daily data to match biweekly short interest dates.")
        return None
    
    X = data.drop(columns=[target_series.name])
    y = data[target_series.name]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse}")

    return model

# We need a function that is going to save the model
def save_model(model, model_path="xgboost_model.json"):
    model.save_model(model_path)

# We need a function that is going to load the model
def load_model(model_path="xgboost_model.json"):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

# We need a function that is going to make predictions with the model
def make_predictions(model, features_df):
    predictions = model.predict(features_df)
    return predictions

# We now have all the pieces to create a full pipeline, now we want to create the full user flow
# for a command-line based application

if __name__ == "__main__":
    # Encapsulate everything in a try and except block to wipe out any generated files if there is an error
    max_tries =  5
    cur_tries = 0
    # Start by asking the user for the set of stocks they want to use
    stock_list = input("Enter a comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL): ").split(',')
    stock_list = [s.strip().upper() for s in stock_list]
    # Ask which stock they want to predict short interest for by printing the list of stocks which we 
    # can find in the short_data directory
    # print("Stocks available for short interest prediction:")
    # short_stocks = os.listdir("short_data")
    # short_stocks = [s.split('_')[0] for s in short_stocks if s.endswith('_SI.csv')]
    # print(", ".join(short_stocks))
    short_stock = input("Enter the stock symbol for which you want to predict short interest (e.g., TSLA): ").strip().upper()
    # Now we need to load in the short interest data for this stock
    short_df = get_FINRA_short_data(short_stock)
    short_plot = f"in_images/{short_stock}_short_interest.png"
    # Now we need to get the daily stock data for each stock in the stock list
    df_dict = {}
    # Get a list of the stocks that we couldn't get data for
    failed_stocks = []
    for stock in tqdm(stock_list):
        # printing the stock we are processing will cause the tqdm bar to break, so we will print a changing line
        tqdm.write(f"Processing stock: {stock}")
        try:
            stock_df = get_daily_stock_data(stock)
            process_stock_data(stock_df, stock)
            df_dict[stock] = stock_df
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            failed_stocks.append(stock)
    # Remove any stocks that failed from the stock list
    stock_list = [s for s in stock_list if s not in failed_stocks]
    if not stock_list:
        print("No valid stocks to process. Exiting.")
        exit(1)
    last_error = ""
    while cur_tries < max_tries:
        try:
            cur_tries += 1
            if last_error != "":
                last_error = f"Previous error that was generated. If this helps improve your response, then please pay attention to it: {last_error}"
            claude_payload = create_claude_payload(stock_list, short_stock, short_plot, last_error=last_error)
            # Now we need to call the claude API
            print("Calling Claude API...")
            claude_response = call_claude_api(claude_payload)
            if claude_response:
                print("\nParsing Claude response...")
                code_block = parse_response(claude_response)
                if code_block:
                    print("Creating engineered features...")
                    # We want to merge all the dataframes in df_dict into a single dataframe with columns prefixed by the stock symbol
                    features_df = pd.DataFrame()
                    for stock, df in df_dict.items():
                        df = df.add_prefix(f"{stock}_")
                        # make sure the date index is also a column
                        if features_df.empty:
                            features_df = df
                        else:
                            features_df = features_df.join(df, how='outer')
                    # Now we need to create the engineered features
                    # Print the columns of features_df
                    print("Columns in master dataframe:")
                    print(features_df.columns)
                    features_df = create_engineered_features(features_df)
                    if not features_df.empty:
                        # If we got here, then we have features to train the model
                        # print the columns of features_df
                        print("Engineered features:")
                        print(features_df.columns)
                        # print the info of features_df
                        print("Feature dataframe info:")
                        print(features_df.info())
                        # I want to make sure the short interest dataframe is not empty
                        print(short_df.head())
                        print("Training XGBoost model...")
                        model = train_xgboost_model(features_df, short_df['short_interest'])
                        print("Saving model...")
                        save_model(model)
                        print("Pipeline complete.")
                    else:
                        print("No features were generated.")
                else:
                    print("Failed to parse Claude response.")
            else:
                print("Claude API call failed.")
            # Finally, let's see if they want to make predictions with the model
            make_pred = input("Do you want to make predictions with the trained model? (yes/no): ").strip().lower()
            if make_pred == 'yes':
                model = load_model()
                predictions = make_predictions(model, features_df)
                print("Predictions:")
                print(predictions)
                # Generate a plot of the predictions with error bars
                plt.figure(figsize=(10, 6))
                plt.plot(short_df.index, short_df['short_interest'], label='Actual Short Interest', color='blue')
                plt.plot(short_df.index, predictions, label='Predicted Short Interest', color='orange')
                plt.fill_between(short_df.index, predictions * 0.9, predictions * 1.1, color='orange', alpha=0.2, label='Prediction Interval (±10%)')
                plt.title(f"{short_stock} - Short Interest Prediction")
                plt.xlabel("Date")
                plt.ylabel("Short Interest")
                plt.legend()
                plt.grid()
                plt.savefig(f"{short_stock}_short_interest_prediction.png")
                plt.show()
            # Clear out the generated features file
            clear_generated_features()
            # Clear out the images and json files
            clear_images_and_json(stock_list + [short_stock])
            break
        except Exception as e:
            # I need to know the line number of the error, so I will import traceback
            import traceback
            traceback.print_exc()
            #clear_generated_features()
            
            print("Cleanup complete.")
            print(f"Error in pipeline: {e}")
            if cur_tries < max_tries:
                last_error = str(e)
                print(f"Retrying... ({cur_tries}/{max_tries})")
            else:
                clear_images_and_json(stock_list + [short_stock])
                clear_generated_features()
                print("Max retries reached. Exiting.")
                break





