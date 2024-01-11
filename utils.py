import ccxt
import math
import pandas as pd
import numpy as np
import urllib
import json
from typing import List
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
from functools import reduce
from plotly.subplots import make_subplots
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
from scipy.stats.mstats import gmean
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
# adjust the memory usage of the plots
plt.rcParams['figure.dpi'] = 80

# Data Cleaning
def get_all_symbols() -> List:
    response = urllib.request.urlopen(
        "https://api.binance.us/api/v3/exchangeInfo").read()
    return list(
        map(lambda symbol: symbol['symbol'], json.loads(response)['symbols']))

def fetch_data_binance(coin: str, timeframe: str, start: str, stop: str) -> pd.DataFrame:
    print(f'fetching {coin} ...')
    try:
        exchange = ccxt.binance()
        start_str = start
        stop_str = stop
        start = exchange.parse8601(start + '00:00:00')
        stop = exchange.parse8601(stop + '00:00:00')

        delta = {'1m': 60000000, '5m': 60000000*5, '1h': 60000000*60}[timeframe]
        segments = []
        while start < stop:
            data = exchange.fetch_ohlcv(coin, timeframe, start, 1000)
            segments.append(pd.DataFrame(data))
            start+=delta

        total = pd.concat(segments)
        total.drop_duplicates(inplace=True)
        total.columns = list('tohlcv')
        print('saving file')
        total.to_csv(f'/home/jd/{coin}_{timeframe}_{start_str}_{stop_str}.csv', index=False)

        return total
    except Exception as e:
        print('no coin', e)


def load_and_clean_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df['Log_close'] = np.log(df['c'])
    df['time'] = df['t'].apply(lambda x: dt.datetime.utcfromtimestamp(x / 1e3))
    df.set_index('time', inplace=True)
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
    df.drop(columns=['t'], inplace=True)
    df.index.name = 'Time'
    df['5-Min Return'] = df['Close'].pct_change()

    return df

# SSD computation
def compute_ssd(Crypto_Price: pd.DataFrame, crypto: str, use_log: bool = False) -> float:
    btc_p = Crypto_Price['BTC Close Price']
    crypto_p = Crypto_Price[crypto.upper() + ' Close Price']

    if use_log:
        btc_p = np.log(btc_p)
        crypto_p = np.log(crypto_p)

    btc_p = (btc_p - np.mean(btc_p)) / np.std(btc_p)
    crypto_p = (crypto_p - np.mean(crypto_p)) / np.std(crypto_p)

    ssd = np.sum((btc_p - crypto_p) ** 2)
    return ssd

# Stop loss computation
def optimal_holding_duration(spread: pd.Series) -> int:
    z_lag = np.roll(spread, 1)
    z_lag[0] = 0
    z_ret = spread - z_lag
    z_ret[0] = 0
    #adds intercept terms to X variable for regression
    z_lag2 = sm.add_constant(z_lag)

    model = sm.OLS(z_ret,z_lag2)
    res = model.fit()
    duration = -np.log(2) / res.params[1]
    return duration

def compute_holding(position: pd.Series, signals: pd.Series) -> pd.Series:
    array = []
    prev_signal, crt_duration = -9, 0
    for pos, sig in zip(position, signals):
        if sig == prev_signal and pos:
            crt_duration += 1
        else:
            crt_duration = 1 if pos else 0
        prev_signal = sig
        array.append(crt_duration)

    return array



# Main simulation function here
def simulation(Crypto_Price, crypto: str, window: int, thresh: float, capital: int, holding_duration: int, fee: float=0, num_exchange: int =15, cost_of_borrow: float=0) -> pd.DataFrame:
    pr_df = pd.DataFrame(Crypto_Price['BTC Close Price']/Crypto_Price[crypto +' Close Price'] ,columns = ['BTC vs '+ crypto])
    pr_df['BTC Close Price'] = Crypto_Price['BTC Close Price']
    pr_df['BTC Volume'] = Crypto_Price['BTC Volume']
    pr_df[f'{crypto} Volume'] = Crypto_Price[crypto +' Volume']
    pr_df[crypto +' Close Price'] = Crypto_Price[crypto +' Close Price']
    pr_df[str(window) + 'h mean'] = pr_df['BTC vs '+ crypto].ewm(span = window,min_periods = window).mean()
    pr_df[str(window) + 'h std'] = pr_df['BTC vs '+ crypto].ewm(span = window,min_periods = window).std()
    upper = pr_df[str(window) + 'h mean'] + thresh * pr_df[str(window) + 'h std']
    lower = pr_df[str(window) + 'h mean'] - thresh * pr_df[str(window) + 'h std']
    pr_df['signal'] = np.where(pr_df['BTC vs '+crypto] < lower, 1, np.where(pr_df['BTC vs '+crypto] > upper, -1, 0))
    df = pr_df.copy()

    df['BTC target pos'] = np.where(pr_df['signal'] == 1, capital / 2 / df['BTC Close Price'], (np.where(pr_df['signal'] ==-1, - capital / 2 / df['BTC Close Price'], 0)))
    df[f'{crypto} target pos'] = np.where(pr_df['signal'] == 1, - capital / 2 /df[f'{crypto} Close Price'], (np.where(pr_df['signal'] ==-1, capital / 2 /df[f'{crypto} Close Price'], 0)))

    # time based stop-loss
    df['holding_time'] = compute_holding(df['BTC target pos'], df['signal'])
    df['BTC target pos'] = np.where(df['holding_time'] >= holding_duration, 0, df['BTC target pos'])
    df[f'{crypto} target pos'] = np.where(df['holding_time'] >= holding_duration, 0, df[f'{crypto} target pos'])

    df['BTC trade'] = df['BTC target pos'].diff().fillna(df['BTC target pos'])
    df[f'{crypto} trade'] = df[f'{crypto} target pos'].diff().fillna(df[f'{crypto} target pos'])

    df['BTC cash'] = (-1 * df['BTC trade'] * df['BTC Close Price']).cumsum()
    df[f'{crypto} cash'] = (-1 * df[f'{crypto} trade'] * df[f'{crypto} Close Price']).cumsum()

    #df['slippage'] = df['BTC trade'].abs() *  df['BTC Close Price'] * slippage_gen(Crypto_Price) + df[f'{crypto} trade'].abs()  * df[f'{crypto} Close Price']*slippage_gen(Crypto_Price, crypto)
    df['slippage_BTC'] =(1/num_exchange * df['BTC trade'].abs() / df['BTC Volume'].rolling(50).mean())**2 * 0.1 * df['BTC Close Price']
    df['slippage_'+crypto] = (1/num_exchange * df[f'{crypto} trade'].abs() / df[f'{crypto} Volume'].rolling(50).mean())**2 * 0.1 * df[f'{crypto} Close Price']
    df['slippage'] = df['slippage_BTC'] + df['slippage_'+crypto]

    df['fee'] = (df['BTC trade'].abs() *  df['BTC Close Price'] + df[f'{crypto} trade'].abs()  * df[f'{crypto} Close Price']) * fee
    df['BTC pnl'] = df['BTC target pos'] * df['BTC Close Price'] + df['BTC cash']
    df[f'{crypto} pnl'] = df[f'{crypto} target pos'] * df[f'{crypto} Close Price'] + df[f'{crypto} cash']

    df['Revenue'] = df['BTC pnl'] + df[f'{crypto} pnl']
    #df['PNL'] = df['Revenue'] - df['fee'].cumsum()
    df['PNL'] = df['Revenue'] -  df['fee'].cumsum()  - df['slippage'].cumsum()
    return df

def slippage_gen(Crypto_Price, crypto:str = 'BTC') -> pd.Series:
    """
    We define the typical slippage range to be 4 to 10 bps. The actual slippage will be randomly generated through
    a random number generator and be adjusted/scaled by the vol regime. 
    
    Use past 100 data points as baseline reference. v_100
    Use recent 10 data points as vol indicator. v_10
    Then the slippage epsilon is scaled by v_10 / v_100
    """

    v_100 = Crypto_Price[f'{crypto} Close Price'].rolling(window = 100).std()
    v_10 = Crypto_Price[f'{crypto} Close Price'].rolling(window = 10).std()
    multiplier = v_10/v_100
    
    epsilon_series = pd.Series(np.random.randint(4, 11, size=multiplier.size), index = multiplier.index)
    
    return epsilon_series * multiplier/10000

def optimize_weekly_rebalance(pnl):
    ret = pnl.diff() / 10_000_1000
    n = ret.shape[1]  # num of assets
    # default weights are just 1/num_assets
    # Store every weight
    w = [np.array([n * [1 / n]])]

    optimized_ret = list()
    return_array = ret.to_numpy()
    # We loosely assume there are 7 * 24 / 2 = 84 timestamps in a week

    period = 84 * 2
    for start_idx in range(0, len(return_array), period):
        end_index = start_idx + period
        training_data = return_array[start_idx:end_index]
        # Compute the covariance matrix
        cov_matrix = np.cov(training_data.T)
        # Compute the tangency weights
        try:
            weights = tangency_portfolio_rfr(training_data.mean(axis=0), cov_matrix)
            if np.isnan(weights).any():
                weights = np.array([n * [1 / n]]).T
                weights = weights[:, 0]
        except np.linalg.LinAlgError:
            weights = np.array([n * [1 / n]]).T
            weights = weights[:, 0]
        # Store the weights
        w.append(weights)
        # apply the weights to subsequent weekly returns
        # print(pnl[start_idx: end_index] @ weights)
        optimized_ret.extend(training_data @ weights * 10_000_000)

    return w, optimized_ret


# Funtion to compute portfolio Sharpe Ratio
def sharpe_ratio(weights, returns):
    portfolio_return = np.dot(weights, returns.mean())
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    sharpe_ratio = portfolio_return / portfolio_std
    return sharpe_ratio


# Compute tangency weights based on given data 
def tangency_portfolio_rfr(asset_return, cov_matrix, cov_diagnolize=False):
    """
        Returns the tangency portfolio weights in a (1 x n) vector
        Inputs:
            asset_return - return for each asset (n x 1) Vector
            cov_matrix = nxn covariance matrix for the assets
    """
    if cov_diagnolize:
        asset_cov = np.diag(np.diag(cov_matrix))
    else:
        asset_cov = np.array(cov_matrix)
    inverted_cov = np.linalg.inv(asset_cov)
    one_vector = np.ones(len(cov_matrix))

    den = (one_vector @ inverted_cov) @ (asset_return)
    num = inverted_cov @ asset_return
    return (1 / den) * num

# Get performance summary matrix
def performance_summary(return_data):
    summary_stats = return_data.mean().to_frame('Mean Return')
    summary_stats['Volatility'] = return_data.std()
    summary_stats['Sharpe Ratio'] = summary_stats['Mean Return'] / summary_stats['Volatility']

    summary_stats['Skewness'] = return_data.skew()
    summary_stats['Excess Kurtosis'] = return_data.kurtosis()
    summary_stats['VaR (0.05)'] = return_data.quantile(.05, axis=0)
    summary_stats['CVaR (0.05)'] = return_data[return_data <= return_data.quantile(.05, axis=0)].mean()
    summary_stats['Min'] = return_data.min()
    summary_stats['Max'] = return_data.max()

    return summary_stats
#compute max drawdown
def MaxDrawdown(nv_list):
    i = np.argmax((np.maximum.accumulate(nv_list) - nv_list) / np.maximum.accumulate(nv_list))
    if i == 0:
        return 0
    j = np.argmax(nv_list[:i])
    return (nv_list[j] - nv_list[i]) / (nv_list[j])

#merge all stats
def perform_statistics(return_all,rf=0,freq='D'):
    multiply = {'D':250,'W':52,'M':12,'Q':4,'Y':1,'H':6000}
    index_metric = ['Annual Return %', 'Sharpe Ratio', 'Sortino', 'Omega', 'Volatility %',
                    'semi-variance %', 'VaR(30days) %','Max drawdwon %','return/Max drawdwon','positive rate %',
                    '1 year return (rolling) %','3 year return (rolling) %','5 year return (rolling) %','1 year rolling VaR(5%) %']
    result = pd.DataFrame(columns=return_all.columns,index=index_metric)
    for i in return_all.columns:
    
        return_s =  return_all[i].dropna()
        return_s.index = pd.DatetimeIndex(return_s.index)
        ret = gmean(return_s.fillna(0)+1)**multiply[freq] -1
        vol = return_s.std() * np.sqrt(multiply[freq])
        sharpe = (ret-rf) / vol
        return_s_monthly=return_s.resample('M').sum()
        odds=return_s_monthly[return_s_monthly>0].count()/return_s_monthly[return_s_monthly!=0].count()

        ret_ts_1y=(return_s+1).map(lambda x:math.log(x)).rolling(multiply[freq]).sum().map(lambda x: math.e**x)-1
        ret_ts_3y=(return_s+1).map(lambda x:math.log(x)).rolling(multiply[freq]*3).sum().map(lambda x: math.e**x)-1
        ret_ts_5y=(return_s+1).map(lambda x:math.log(x)).rolling(multiply[freq]*5).sum().map(lambda x: math.e**x)-1
        ret_1y=ret_ts_1y.dropna().mean()
        ret_3y=ret_ts_3y.dropna().mean()
        ret_5y=ret_ts_5y.dropna().mean()
        VaR_1y=ret_ts_1y.dropna().quantile(0.05)
                               
        
        return_prf = pd.DataFrame()
        rf = 0  
        return_prf["ex_ret"] = return_s - rf 
        return_prf["neg_ex_ret"] = return_prf["ex_ret"][return_prf["ex_ret"] < 0]  
        return_prf["pos_ex_ret"] = return_prf["ex_ret"][return_prf["ex_ret"] > 0]  
        return_prf = return_prf.fillna(0)  
    
        
        semi_var = np.sqrt(np.sum(return_prf["neg_ex_ret"] ** 2) / len(return_prf["neg_ex_ret"]))
        if return_prf["neg_ex_ret"].sum() == 0: 
            sortino = np.nan
            omega = np.nan
        else:
            sortino = return_prf["ex_ret"].mean() / semi_var
            omega = return_prf["pos_ex_ret"].sum() / - return_prf["neg_ex_ret"].sum()
    
        
        rank_ratio = return_s[1:].sort_values(ascending=True)
        var = np.percentile(rank_ratio, 5, interpolation='lower') * np.sqrt(multiply[freq]/12)
    
        max_drawback = MaxDrawdown((1 + return_s).cumprod())
    
        result[i] = np.array([ret * 100, sharpe, sortino, omega, vol * 100, semi_var * 100, var * 100, max_drawback * 100 ,ret/max_drawback,odds*100,ret_1y*100,ret_3y*100,ret_5y*100,VaR_1y*100])
    
    return result.round(2)


def adf_cointegration_test(dataframe: pd.DataFrame, asset_name_1: str, asset_name_2: str):
    # Step 1: Perform Augmented Dickey-Fuller test for each series
    series1 = np.log(dataframe[f'{asset_name_1} Close Price'])
    series2 = np.log(dataframe[f'{asset_name_2} Close Price'])
    adf_result_1 = adfuller(series1)
    adf_result_2 = adfuller(series2)

    # Step 2: Check if both series are individually stationary
    if adf_result_1[1] > 0.05 or adf_result_2[1] > 0.05:
        print("Series are not individually stationary. Cointegration test may not be valid.")
        #return None

    # Step 3: Perform regression on the levels (first differences)
    diff_series1 = np.diff(series1)
    diff_series2 = np.diff(series2)

    # Include a constant term in the regression
    X = np.vstack([np.ones_like(diff_series1), diff_series1]).T
    beta_hat = np.linalg.lstsq(X, diff_series2, rcond=None)[0]

    # Step 4: Check if the residuals are stationary
    residuals = diff_series2 - X.dot(beta_hat)
    adf_result_residuals = adfuller(residuals)

    # Step 5: Check if residuals are stationary
    if adf_result_residuals[1] <= 0.05:
        print("Cointegration test passed. The series are cointegrated.")
        return beta_hat
    else:
        print("Cointegration test failed. The series are not cointegrated.")
        return None