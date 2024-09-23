import pandas as pd
import time
import os
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

def start_training_date(start_forecast, training_months):
# Function to calculate start of training date from forecasting date and number of months required
    start_training_dt = pd.to_datetime(start_forecast) - pd.DateOffset(months=training_months)
    return start_training_dt


def end_training_date(start_forecast):
# Function to calculate end of training date from forecasting date
    end_forecast_dt = pd.to_datetime(start_forecast) - pd.DateOffset(months=1)
    return end_forecast_dt


def financial_year_start(month_date):
# Function to calculate the start of the financial year
    if pd.Timestamp(month_date).month < 4: # if month_date is before April, return 1st April previous calendar year
        financial_year_start = pd.Timestamp(year=pd.Timestamp(month_date).year - 1, month=4, day=1)
    else: # return 1st April current calendar year
        financial_year_start = pd.Timestamp(year=pd.Timestamp(month_date).year, month=4, day=1)
    return financial_year_start


def financial_year_end(month_date):
# Function to calculate the start of the financial year
    if pd.Timestamp(month_date).month >= 4: # if month_date is after April, return 1st March next calendar year
        financial_year_end = pd.Timestamp(year=pd.Timestamp(month_date).year + 1, month=3, day=1)
    else: # return 1st March current calendar year
        financial_year_end = pd.Timestamp(year=pd.Timestamp(month_date).year, month=3, day=1)
    return financial_year_end


def print_time_summary(start_time, end_time):
# Function to show start and end times and elapsed of a process running in the notebook (e.g. AutoARIMA processing)
    
    # Convert to a readable format
    start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    
    # Print the start and end times, and the elapsed time
    print(f"Start Time: {start_time_readable}")
    print(f"End Time: {end_time_readable}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")


def calculate_price_new(row):
# Function used in dmd data import to sort current from previous prices
    
    # Check if 'pricedt' is NaT or if 'month' is greater than or equal to 'pricedt'
    if pd.isna(row['priceinfo_pricedt']) or row['month'] >= row['priceinfo_pricedt']:
        return row['priceinfo_price']
    else:
        return row['priceinfo_price_prev']


def rank_and_filter_prices(df, rank_column='pc_price_per_unit', price_column='price_per_unit', rank_method='first', ascending=False):
# Function to create the higest price-per-unit for each each BNF code, to ensure no duplication due to pack sizes (e.g. paracetamol 500mg tablets 32 and 100)

    # Create a ranking column for price concession if it exists, otherwise use the actual price
    df['rank_value'] = df[rank_column].fillna(df[price_column])
    
    # Create the ranking column based on the rank_value
    df['ppu_rank'] = df.groupby(['month', 'bnf_code'])['rank_value'].rank(method=rank_method, ascending=ascending)
    
    # Filter the DataFrame to include only the top-ranked rows
    filtered_df = df[df['ppu_rank'] == 1][['month', 'bnf_code', 'dt_category', price_column, rank_column]]
    
    return filtered_df


def process_dmd_data(dmd_raw_df, start_forecast, end_forecast):
# Function to create monthly price list from dm+d data, to enable price calculations for each month of data in the forecast period
    
    # Generate the date range for the forecast period
    date_range = pd.date_range(start=start_forecast, end=end_forecast, freq='MS')
    
    # Create a DataFrame with the date range
    date_range_df = pd.DataFrame(date_range, columns=['month'])
    
    # Add a key to perform a cross join
    date_range_df['key'] = 1
    dmd_raw_df['key'] = 1
    
    # Perform a cross join
    combined_df = pd.merge(date_range_df, dmd_raw_df, on='key').drop('key', axis=1)
    
    # Ensure 'priceinfo_pricedt' is in datetime format
    combined_df['priceinfo_pricedt'] = pd.to_datetime(combined_df['priceinfo_pricedt'])
    
    # Sort by 'bnf_code' and 'month' to prepare for forward fill
    combined_df.sort_values(by=['bnf_code', 'month'], inplace=True)
    
    # Forward fill price information
    combined_df['priceinfo_price'] = combined_df.groupby('bnf_code')['priceinfo_price'].ffill()
    combined_df['priceinfo_price_prev'] = combined_df.groupby('bnf_code')['priceinfo_price_prev'].ffill()
    
    # Calculate the correct price based on the conditions
    combined_df['price_pence'] = combined_df.apply(
        lambda row: row['priceinfo_price'] if pd.isna(row['priceinfo_pricedt']) or row['month'] >= row['priceinfo_pricedt']
        else row['priceinfo_price_prev'], axis=1
    )

    # Calculate price per unit
    combined_df['price_per_unit'] = combined_df['price_pence'] / combined_df['qtyval']

    # Rank price per unit
    combined_df['row_number'] = combined_df.groupby(['month', 'bnf_code'])['price_per_unit'].rank(method='first', ascending=False)
    
    # Filter to get the top-ranked price per unit per month and BNF code
    result_df = combined_df[combined_df['row_number'] == 1][['month', 'bnf_code', 'price_per_unit']]

    return result_df


def merge_and_update_prices(dmd_price_df, dt_price_df):
# Function to merge drug tariff and dm+d price lists
    
    # Merge datasets
    merged_price_df = dmd_price_df.merge(dt_price_df[['month', 'bnf_code']], on=['month', 'bnf_code'], how='left', indicator=True)
    
    # Create dataset where only dmd bnf_codes exist
    dmd_unique_df = merged_price_df[merged_price_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # Merge dmd_bnf_codes only into dt dataset to create full monthly price list
    full_price_df = pd.concat([dt_price_df, dmd_unique_df], ignore_index=True)
    
    # Convert columns to float
    full_price_df['price_per_unit'] = full_price_df['price_per_unit'].astype(float)
    full_price_df['pc_price_per_unit'] = full_price_df['pc_price_per_unit'].astype(float)
    
    return full_price_df


def fill_missing_dates(test_full_price_df, end_forecast):
# Function to create price list for rest of financial year, taking price from latest month
    
    # Convert end_forecast to a Timestamp
    end_of_financial_year = pd.Timestamp(end_forecast)
    
    # Create a full date range for the entire dataset
    all_dates = pd.date_range(start=test_full_price_df['month'].min(), end=end_of_financial_year, freq='MS')
    
    # Create a DataFrame for the full date range
    full_date_range_df = pd.DataFrame({'month': all_dates})
    
    # Create a DataFrame with all possible month-bnf_code combinations
    bnf_codes = test_full_price_df['bnf_code'].unique()
    all_combinations = pd.MultiIndex.from_product([all_dates, bnf_codes], names=['month', 'bnf_code']).to_frame(index=False)
    
    # Merge full combinations with the original data to get all possible rows
    merged_df = all_combinations.merge(test_full_price_df, on=['month', 'bnf_code'], how='left')
    
    # Forward fill missing prices
    merged_df['price_per_unit'] = merged_df.groupby('bnf_code')['price_per_unit'].ffill()
    merged_df['pc_price_per_unit'] = merged_df.groupby('bnf_code')['pc_price_per_unit'].ffill()
    
    return merged_df


def filter_top_bnf_codes(rx_df, top_x_percent):
# Function to filter BNF codes based on top x% of items AND actual cost
    
    # Group by 'bnf_code' and aggregate sum of 'items' and 'actual_cost'
    grouped_rx_df = rx_df.groupby('bnf_code').agg({'items': 'sum', 'actual_cost': 'sum'}).reset_index()
    
    # Sort by 'items' to get the cumulative sum for items
    grouped_rx_df = grouped_rx_df.sort_values('items', ascending=False)
    
    # Calculate the cumulative sum and the total sum for items
    grouped_rx_df['cumulative_items'] = grouped_rx_df['items'].cumsum()
    total_items = grouped_rx_df['items'].sum()
    
    # Calculate threshold value and find the bnf_codes that contribute to the top x% of items
    threshold_items = total_items * (top_x_percent / 100)
    top_items_bnf_codes = grouped_rx_df[grouped_rx_df['cumulative_items'] <= threshold_items]['bnf_code']
    
    # Repeat the same process for 'actual_cost'
    grouped_rx_df = grouped_rx_df.sort_values('actual_cost', ascending=False)
    grouped_rx_df['cumulative_actual_cost'] = grouped_rx_df['actual_cost'].cumsum()
    total_actual_cost = grouped_rx_df['actual_cost'].sum()
    threshold_actual_cost = total_actual_cost * (top_x_percent / 100)
    top_cost_bnf_codes = grouped_rx_df[grouped_rx_df['cumulative_actual_cost'] <= threshold_actual_cost]['bnf_code']
    
    # Combine both sets of bnf_codes
    top_bnf_codes = set(top_items_bnf_codes).union(set(top_cost_bnf_codes))
    
    # Filter the original DataFrame for these top bnf_codes
    topx_rx_df = rx_df[rx_df['bnf_code'].isin(top_bnf_codes)]
    
    # Filter the original DataFrame for non-top bnf_codes
    non_topx_rx_df = rx_df[~rx_df['bnf_code'].isin(top_bnf_codes)]
    
    print(f'Number of BNF codes in top {top_x_percent}% items or spend: {len(top_bnf_codes)}')
    
    return topx_rx_df, non_topx_rx_df, top_bnf_codes


def fill_missing_combinations(final_df, min_month, max_month):
    # Create a copy of the relevant columns
    missing_df = final_df[['ds', 'unique_id', 'y']].copy()
    
    # Create a complete list of months and bnf_codes
    complete_months = pd.date_range(start=min_month, end=max_month, freq='MS').strftime('%Y-%m').tolist()
    complete_bnf_codes = missing_df['unique_id'].unique().tolist()
    
    # Create a MultiIndex from the complete list of months and bnf_codes
    multi_index = pd.MultiIndex.from_product([complete_months, complete_bnf_codes], names=['ds', 'unique_id'])
    
    # Reindex the original DataFrame to fill in missing combinations
    df = missing_df.set_index(['ds', 'unique_id']).reindex(multi_index, fill_value=0).reset_index()
    
    return df


def nadp_adjuster(df, nadp_df):
#Function to adjust tariff prices to actual cost, using NADP pre-April 2024, then fixed discount rates from April 2024
    
    # Preprocess nadp_df: Set Date as the index and convert percentage to float
    nadp_df.set_index('month', inplace=True)
    nadp_df['nadp'] = nadp_df['nadp'].astype(float) / 100
    
    # Ensure the month column in df is datetime
    df['month'] = pd.to_datetime(df['month'])
    
    # Condition for dates on or after where price concessions are zero discount and discount calculation changes
    condition_post_april = df['month'] >= pd.to_datetime('2024-04-01') # NADP changes
    condition_pc_nadp_change = df['month'] >= pd.to_datetime('2023-04-01') # Price concession changes

    # Update actual cost for dates from April 2024
    df.loc[condition_post_april & df['dt_category'].isin([1, 11]), 'price_per_unit'] *= 0.8 # DT category A and M = 20% discount
    df.loc[condition_post_april & df['dt_category'].isnull() & df['bnf_code'].str.startswith(('19', '20')), 'price_per_unit'] *= 0.9015 # Appliance (by BNF code) = 10% discount
    df.loc[condition_post_april & df['dt_category'].isnull() & ~df['bnf_code'].str.startswith(('19', '20')), 'price_per_unit'] *= 0.95 # No category stated, and not an appliance (by BNF code) = 5% discount
    df.loc[condition_post_april & df['dt_category'].notnull() & ~df['dt_category'].isin([1, 11]), 'price_per_unit'] *= 0.95 # DT category not in A and M = 5% discount

    # Apply NADP values for dates before April 2024
    nadp_mapping = nadp_df['nadp'].to_dict()
    df.loc[~condition_post_april, 'price_per_unit'] *= (1 - df.loc[~condition_post_april, 'month'].map(nadp_mapping).fillna(1.0))

    #Apply NADP values for price concessions (only for before April 2023)
    df.loc[~condition_pc_nadp_change, 'pc_price_per_unit'] *= (1 - df.loc[~condition_pc_nadp_change, 'month'].map(nadp_mapping).fillna(1.0))    


def create_arima_bnf_codes(rx_df, full_price_df, full_price_bnf_codes, current_fy, end_training, top_x_percent, multiplier_limit):
# Function to find top x by both items and actual cost, then removing those without a cost or outside of expected PPU multiplier limits (both x and 1/x)
# Creates two dfs, one for individual AutoARIMA, and one with exlcuded codes *WHAT TO DO*
    
    # Group by 'bnf_code' and aggregate sum of 'items' and 'actual_cost'
    grouped_rx_df = rx_df.groupby('bnf_code').agg({'items': 'sum', 'actual_cost': 'sum'}).reset_index()
    
    # Calculate top bnf_codes by items
    grouped_rx_df = grouped_rx_df.sort_values('items', ascending=False)
    grouped_rx_df['cumulative_items'] = grouped_rx_df['items'].cumsum()
    total_items = grouped_rx_df['items'].sum()
    threshold_items = total_items * (top_x_percent / 100)
    top_items_bnf_codes = grouped_rx_df[grouped_rx_df['cumulative_items'] <= threshold_items]['bnf_code']

    # Calculate top bnf_codes by actual cost
    grouped_rx_df = grouped_rx_df.sort_values('actual_cost', ascending=False)
    grouped_rx_df['cumulative_actual_cost'] = grouped_rx_df['actual_cost'].cumsum()
    total_actual_cost = grouped_rx_df['actual_cost'].sum()
    threshold_actual_cost = total_actual_cost * (top_x_percent / 100)
    top_cost_bnf_codes = grouped_rx_df[grouped_rx_df['cumulative_actual_cost'] <= threshold_actual_cost]['bnf_code']

    # Combine bnf_codes from items and actual cost
    top_bnf_codes = set(top_items_bnf_codes).union(set(top_cost_bnf_codes))

    # Remove any bnf_codes not present in the full price list
    top_bnf_codes = [code for code in top_bnf_codes if code in full_price_bnf_codes['bnf_code'].values]

    # Remove BNF codes outside of multiplier limits:
    
    current_rx_ppu_df = (rx_df[(rx_df['month'] >= current_fy) & (rx_df['month'] <= end_training)] # Filter for current financial year and aggregate for average price per unit (PPU)
                         .groupby(['bnf_code', 'bnf_name'], as_index=False)
                         .agg({'quantity': 'sum', 'actual_cost': 'sum'}))

    current_rx_ppu_df['avg_ppu'] = (current_rx_ppu_df['actual_cost'] / current_rx_ppu_df['quantity']) * 100  # Compute average price per unit (PPU) in pence
    median_price_df = full_price_df.groupby('bnf_code', as_index=False).median()[['bnf_code', 'price_per_unit']] #Find median price per unit from the full price list
    merged_ppu_df = pd.merge(current_rx_ppu_df[['bnf_code', 'avg_ppu']], median_price_df, on='bnf_code') #Merge the calculated PPU with the median price per unit
    merged_ppu_df['multiplier'] = merged_ppu_df['avg_ppu'] / merged_ppu_df['price_per_unit'] #Calculate the multiplicated difference between actual and expected PPU
    bnf_filter = (merged_ppu_df['multiplier'] >= multiplier_limit) | (merged_ppu_df['multiplier'] <= 1 / multiplier_limit) #Filter out bnf_codes that have a multiplier outside the set limits
    multiplier_bnf_codes = merged_ppu_df.loc[bnf_filter, 'bnf_code'].tolist() #create list of BNF codes which are outside limits
    top_bnf_codes = [code for code in top_bnf_codes if code not in multiplier_bnf_codes] #Remove codes from top_bnf_codes if outside multiplier limits

    # Create dataframes, splitting prescribing data into Topx and not-topx dfs
    topx_rx_df = rx_df[rx_df['bnf_code'].isin(top_bnf_codes)]
    non_topx_rx_df = rx_df[~rx_df['bnf_code'].isin(top_bnf_codes)]

    # Print the number of BNF codes in the top percentage
    print(f'Number of BNF codes in top {top_x_percent}% items or spend: {len(top_bnf_codes)}')

    # Return the filtered DataFrames
    return topx_rx_df, non_topx_rx_df


def forecast_prescribing_data(df, topx_rx_df, start_training, end_training, horizon, season_length):
# Function to use AutoARIMA to calculate forecasting
    
    # Filter the DataFrame for the training period
    filtered_arima_df = df[(df['ds'] >= start_training) & (df['ds'] <= end_training)]
    filtered_arima_df = filtered_arima_df.sort_values(by=['ds', 'unique_id'])

    # Ensure seasonal data has zeros in other months (e.g. flu vaccines) using fill_missing_combinations function
    training_arima_df = fill_missing_combinations(filtered_arima_df, start_training, end_training)
    
    # Set AutoARIMA as the model and environment variables
    os.environ['NIXTLA_ID_AS_COL'] = '1'  # Unique ID as column, not index
    models = [AutoARIMA(season_length=season_length)]  # AutoARIMA with specified season length
    
    # Initialize the StatsForecast object
    sf = StatsForecast(models=models, freq='MS', n_jobs=-1)  # Monthly frequency ('MS'), use all available processors
    
    # Run AutoARIMA modelling
    forecast_rx_df = sf.forecast(df=training_arima_df, h=horizon, level=[95])

    # Merge with prescribing data
    full_forecast_rx_df = pd.merge(topx_rx_df, forecast_rx_df, left_on=['month', 'bnf_code'], right_on=['ds','unique_id'], how='outer').sort_values(by=['month'])
    
    return full_forecast_rx_df


import matplotlib.pyplot as plt

def plot_forecasts(full_forecast_rx_df, num_codes=50):
    # Step 1: Get unique bnf_codes and limit to first 'num_codes'
    unique_bnf_codes = full_forecast_rx_df['bnf_code'].unique()
    first_x_bnf_codes = unique_bnf_codes[:num_codes]
    
    # Step 2: Loop through the first 'num_codes' bnf_codes
    for code in first_x_bnf_codes:
        # Filter the DataFrame for the current bnf_code
        filtered_df = full_forecast_rx_df[full_forecast_rx_df['bnf_code'] == code]
        
        # Step 3: Plot month vs quantity and forecast (AutoARIMA)
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_df['month'], filtered_df['quantity'], label='Actual Quantity')
        plt.plot(filtered_df['month'], filtered_df['AutoARIMA'], label='Forecasted Quantity')
        
        # Plot the confidence intervals (95% CI)
        plt.fill_between(filtered_df['month'], filtered_df['AutoARIMA-lo-95'], filtered_df['AutoARIMA-hi-95'], 
                         color='gray', alpha=0.3, label='95% CI')
        
        # Set plot limits, labels, and title
        plt.ylim(bottom=0)
        plt.xlabel('Month')
        plt.ylabel('Quantity')
        plt.title(f'Forecast for BNF Code: {code}')
        
        # Add a legend
        plt.legend()
        
        # Show the plot
        plt.show()