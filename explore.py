import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', size=13)
plt.rc('figure', figsize=(13,7))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import env
import wrangle

def plot_variable_pairs(df):
    g = sns.PairGrid(df)
    g.map_diag(sns.distplot)
    g.map_offdiag(sns.regplot)
    return g

def months_to_years(df):
    import math
    df['tenure_years'] = df.tenure.apply(lambda x: math.floor(x/12))
    return df

def plot_categorical_and_continuous_vars(df, column_cont, column_cat, hue_arg):
    plt.rc('font', size=13)
    plt.rc('figure', figsize=(13,9))
    plt.subplot(311)
    plot1 = sns.boxplot(data=df, y=column_cont, x=column_cat)
    plt.subplot(312)
    plot2 = sns.barplot(data=df, y=column_cont, x=column_cat, hue=hue_arg)
    plt.subplot(313)
    plot3 = sns.swarmplot(data=df, y=column_cont, x=column_cat, hue=hue_arg)
    plt.tight_layout()
    return plot1, plot2, plot3

def plot_categorical_vars(df, column1, column2, normalized_arg):
    ct = pd.crosstab(column1, column2, normalize=normalized_arg)
    htmp= sns.heatmap(ct, cmap='Greens', annot=True, fmt='.1%')
    return htmp