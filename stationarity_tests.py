from statsmodels.tsa.stattools import adfuller, kpss


import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter('ignore', InterpolationWarning)

def adf_test(timeseries, alpha=0.05):
    """
    Perform ADF test for stationarity.
    Returns:
    - bool: True if stationary (reject null hypothesis non stationary), 
            False otherwise.
    """
    dftest = adfuller(timeseries, autolag="AIC")
    test_stat = dftest[0]
    crit_val = dftest[4][f"{int(alpha*100)}%"]
    return test_stat < crit_val

def kpss_test(timeseries, alpha=0.05):
    """
    Perform KPSS test for stationarity.
    Returns:
    - bool: True if stationary (fail to reject null hypothesis stationary),
            False otherwise.
    """
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    test_stat = kpsstest[0]
    crit_val = kpsstest[3][f"{int(alpha*100)}%"]
    return not test_stat > crit_val

def stationarity_test(timeseries, alpha=0.05):
    """
    Combine ADF and KPSS tests for stationarity.

    Case 1: Both tests conclude that the series is not stationary 
            - The series is not stationary
    Case 2: Both tests conclude that the series is stationary 
            - The series is stationary
    Case 3: KPSS indicates stationarity and ADF indicates non-stationarity 
            - The series is trend stationary. Trend needs to be removed to make series strict stationary.
    Case 4: KPSS indicates non-stationarity and ADF indicates stationarity 
            - The series is difference stationary. Differencing is to be used to make series stationary.
    """
    adf_result = adf_test(timeseries, alpha)
    kpss_result = kpss_test(timeseries, alpha)

    if not adf_result and not kpss_result:
        return "Not Stationary"  # Case 1
    elif adf_result and kpss_result:
        return "Stationary"  # Case 2
    elif kpss_result and not adf_result:
        return "Trend Stationary"  # Case 3
    elif not kpss_result and adf_result:
        return "Difference Stationary"  # Case 4