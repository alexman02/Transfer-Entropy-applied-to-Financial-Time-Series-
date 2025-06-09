import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta


def load_and_align_data(file_paths, index_col="Date"):
    dataframes = []
    tickers = {}
    for asset, file_path in file_paths.items():
        df = pd.read_excel(file_path, sheet_name=0)
        df = df.set_index(index_col)
        df.index = pd.to_datetime(df.index)
        tickers[asset] = df.columns.tolist()
        dataframes.append(df)
    # Align datasets
    aligned_data = dataframes[0]
    for df in dataframes[1:]:
        aligned_data = aligned_data.join(df, how="inner")
    # Reverse time
    return aligned_data[::-1], tickers


def drop_short_tickers(data, tickers, max_nan=50):
    short_tickers = data.columns[data.isnull().sum() > max_nan].tolist()
    print("Tickers to drop/replace due to insufficient data:")
    print(short_tickers)

    data = data.drop(columns=short_tickers)
    updated_tickers = {
        asset: [ticker for ticker in ticker_list if ticker not in short_tickers]
        for asset, ticker_list in tickers.items()
    }

    return data, updated_tickers


def compute_log_returns(price_df):
    # Replace zeros with NaN 
    price_df = price_df.replace(0, np.nan)
    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns


def compute_pca(data, n_components=None):
    standardized_data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(standardized_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    return pca, pca_components, explained_variance_ratio


def discretize(contin_data, num_bins=12):
    std = np.std(contin_data)
    bins = std * np.arange(-num_bins//2, num_bins//2 + 1)
    return np.digitize(contin_data, bins)


def pca_and_discretize(log_returns, tickers, plot_explained_var_ratio=False):
    asset_classes_discrete = {}
    explained_variance_ratios = {}
    asset_classes = tickers.keys()
    for asset_class in asset_classes:
        vals = log_returns[tickers[asset_class]]
        pca, pca_components, explained_variance_ratio = compute_pca(vals)
        explained_variance_ratios[asset_class] = explained_variance_ratio
        X = pca_components[:, 0]  # Take first principal component
        discrete_X = discretize(X)
        asset_classes_discrete[asset_class] = discrete_X

    dates = log_returns.index
    asset_classes_discrete = pd.DataFrame(asset_classes_discrete)
    asset_classes_discrete.index = dates
    asset_classes_discrete.index = pd.to_datetime(asset_classes_discrete.index)

    if plot_explained_var_ratio:
        for asset, variance_ratio in explained_variance_ratios.items():
            plt.plot(np.cumsum(variance_ratio), marker="o", label=asset)
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance Ratios")
        plt.legend()
        plt.show()
    return asset_classes_discrete

def discretize_tickers(log_returns, tickers):
    tickers_discrete = {}
    asset_classes = tickers.keys()
    for asset_class in asset_classes:
        for ticker in tickers[asset_class]:
            discrete_ticker = discretize(log_returns[ticker])
            tickers_discrete[ticker] = discrete_ticker

    dates = log_returns.index
    tickers_discrete = pd.DataFrame(tickers_discrete)
    tickers_discrete.index = dates
    tickers_discrete.index = pd.to_datetime(tickers_discrete.index)
    return tickers_discrete


def sliding_window(data, window_size, step_size, start_date=None, end_date=None,
                   window_unit="years", step_unit="years"):
    """
    Generate sliding windows of data for all columns.

    Parameters:
    - data (pd.DataFrame): DataFrame with a datetime index.
    - window_size (int): Size of each sliding window.
    - step_size (int): Step size to slide the window forward.
    - window_unit (str): Unit for the window size (e.g., "years", "months", "days").
    - step_unit (str): Unit for the step size (e.g., "years", "months", "days").

    Returns:
    - windows (list of tuples): Each tuple contains:
         - start_date (pd.Timestamp): Start date of the window.
         - end_date (pd.Timestamp): End date of the window.
         - window_data (pd.DataFrame): The sliced DataFrame for the window.
    """
    windows = []
    if start_date is None:
        start_date = data.index.min()
    if end_date is None:
        end_date = data.index.max()
    curr_date = start_date

    while curr_date + relativedelta(**{window_unit: window_size}) <= end_date:
        # Define the current window using the specified unit
        window_end_date = curr_date + relativedelta(**{window_unit: window_size})
        window_data = data[(data.index >= curr_date) & (data.index < window_end_date)]
        windows.append((curr_date, window_end_date, window_data))

        # Move the window forward by the step size and unit
        curr_date += relativedelta(**{step_unit: step_size})

    return windows

def discretize_equal_width_iqr(contin_data, num_bins=2):
    min_val = contin_data.min()
    max_val = contin_data.max()
    if num_bins == 2:
        med = np.median(contin_data)
        bins = [min_val, med, max_val]
    else:
        q1 = np.percentile(contin_data, 25)
        q3 = np.percentile(contin_data, 75)
        bins = [min_val] + np.linspace(q1, q3, num_bins - 2).tolist() + [max_val]
    # std = np.std(contin_data)
    # bins = std * np.arange(-num_bins//2, num_bins//2 + 1)
    return np.digitize(contin_data, bins)

def discretize_equal_freq_iqr(contin_data, num_bins=2):
    # Use quantiles to create bins
    min_val = contin_data.min()
    max_val = contin_data.max()
    if num_bins == 2:
        med = np.median(contin_data)
        bins = [min_val, med, max_val]
    else:
        q1 = np.percentile(contin_data, 25)
        q3 = np.percentile(contin_data, 75)
        iqr_data = contin_data[(contin_data >= q1) & (contin_data <= q3)]
        quantiles = np.linspace(0, 1, num_bins - 2)
        bins = [min_val] + np.quantile(iqr_data, quantiles).tolist() + [max_val]
    
    return np.digitize(contin_data, bins)

def discretize_equal_freq(contin_data, num_bins=2):
    # Create quantile-based bins so that each bin contains roughly equal number of data points.
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(contin_data, quantiles)
    # Avoid duplicate bin edges in case of ties.
    bins = np.unique(bins)
    discretized = np.digitize(contin_data, bins[1:-1])
        
    return discretized

def discretize_df(dataframe, disc_func = discretize_equal_freq, **kwargs):
    discretized = {}
    columns = dataframe.columns

    for col in columns:
        discrete_col = disc_func(dataframe[col], **kwargs)
        discretized[col] = discrete_col

    dates = dataframe.index
    discretized = pd.DataFrame(discretized)
    discretized.index = pd.to_datetime(dates)
    return discretized

