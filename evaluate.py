import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()

def regression_errors(actual, predicted):
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    MSE = mean_squared_error(actual, predicted)
    SSE = MSE*len(actual)
    ESS = sum((predicted - actual.mean())**2)
    TSS = ESS + SSE
    RMSE = sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(actual):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    df = pd.DataFrame()
    df['y'] = actual
    df['baseline'] = actual.mean()
    
    MSE = mean_squared_error(actual, df.baseline)
    SSE = MSE*len(actual)
    RMSE = sqrt(MSE)
    return SSE, MSE, RMSE

def better_than_baseline(actual, predicted):
    df = pd.DataFrame()
    df['y'] = actual
    df['baseline'] = actual.mean()
    
    baseline_sse = ((df.baseline - actual)**2).sum()
    model_sse = ((predicted - actual)**2).sum()

    if model_sse < baseline_sse:
        return True
    else:
        return False

def model_significance(ols_model):
    r2 = ols_model.rsquared
    f_pval = ols_model.f_pvalue
    return r2, f_pval