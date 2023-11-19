import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from pandas_datareader import data
from scipy.stats import pearsonr
import yfinance as yf
yf.pdr_override()
np.random.seed(5764)

# Q1.
df=pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv")
print("Number of obs: ",len(df))
features=['Sales','AdBudget','GDP']
df_corr_tab = PrettyTable([""]+features)
df_corr_tab.title="Pearson Correlation Matrix for the tute1 dataset"
for i in features:
    each_row=[i]
    for j in features:
        each_row.append(np.corrcoef(df[i],df[j])[0][1].round(2))
    df_corr_tab.add_row(each_row)
print(df_corr_tab)

# Q2.
def pcorr(X,Y,Z):
    Rxy=np.corrcoef(df[X],df[Y])[1][0]
    Rxz=np.corrcoef(df[X],df[Z])[1][0]
    Ryz=np.corrcoef(df[Y],df[Z])[1][0]
    return (Rxy-Rxz*Ryz)/(np.sqrt(1-Rxz**2)*np.sqrt(1-Ryz**2))
df_corr_tab = PrettyTable([""]+features)
df_corr_tab.title="Partial Correlation Matrix for the tute1 dataset"
pcr_sa=round(pcorr('Sales','AdBudget','GDP'),2)
pcr_as=round(pcorr('AdBudget','Sales','GDP'),2)
pcr_ag=round(pcorr('AdBudget','GDP','Sales'),2)
pcr_ga=round(pcorr('GDP','AdBudget','Sales'),2)
pcr_sg=round(pcorr('Sales','GDP','AdBudget'),2)
pcr_gs=round(pcorr('GDP','Sales','AdBudget'),2)
df_corr_tab.add_row(["Sales","-",pcr_sa,pcr_sg])
df_corr_tab.add_row(["AdBudget",pcr_as,"-",pcr_ag])
df_corr_tab.add_row(["GDP",pcr_gs,pcr_ga,"-"])
print(df_corr_tab)

def test_stat_form(x, n, k):
    return x * np.sqrt((n - 2 - k) / (1 - x ** 2))
tab_ccoef = PrettyTable([""]+features)
tab_ccoef.title="T-test values for Correlation coefficients"
for i in features:
    each_row=[i]
    for j in features:
        each_row.append(test_stat_form(np.corrcoef(df[i],df[j])[0][1],len(df),0).round(2))
    tab_ccoef.add_row(each_row)
print(tab_ccoef)

tab_pcoef = PrettyTable([""]+features)
tab_pcoef.title="T-test values for Partial Correlation coefficients"
h_pcr_sa=round(test_stat_form(pcorr('Sales','AdBudget','GDP'),len(df),0),2)
h_pcr_as=round(test_stat_form(pcorr('AdBudget','Sales','GDP'),len(df),0),2)
h_pcr_ag=round(test_stat_form(pcorr('AdBudget','GDP','Sales'),len(df),0),2)
h_pcr_ga=round(test_stat_form(pcorr('GDP','AdBudget','Sales'),len(df),0),2)
h_pcr_sg=round(test_stat_form(pcorr('Sales','GDP','AdBudget'),len(df),0),2)
h_pcr_gs=round(test_stat_form(pcorr('GDP','Sales','AdBudget'),len(df),0),2)
tab_pcoef.add_row(["Sales","-",h_pcr_sa,h_pcr_sg])
tab_pcoef.add_row(["AdBudget",h_pcr_as,"-",h_pcr_ag])
tab_pcoef.add_row(["GDP",h_pcr_gs,h_pcr_ga,"-"])
print(tab_pcoef)


stocks=['AAPL','ORCL','TSLA','IBM','YELP','MSFT']
features=['High','Low','Open','Close','Volume','Adj Close']
df_dict=dict()
for i in stocks:
    df_dict[i]=data.get_data_yahoo(i,start='2000-01-01',end='2023-08-28')

#Q5 & Q6
for f in features:
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    k=0
    for i in range(3):
        for j in range(2):
            axs[i][j].title.set_text(f+" history of "+stocks[k]+" stock")
            axs[i][j].set_xlabel('Year')
            axs[i][j].set_ylabel('USD($)')
            axs[i][j].plot(df_dict[stocks[k]][f])
            axs[i][j].grid()
            k+=1
    fig.tight_layout(pad=1.0)
    plt.show()

#Q7 and Q8
for f in features:
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    k=0
    for i in range(3):
        for j in range(2):
            axs[i][j].title.set_text("Histogram of "+f+" history of "+stocks[k]+" stock")
            axs[i][j].set_xlabel('Frequency')
            axs[i][j].set_xlabel('USD($)')
            axs[i][j].hist(df_dict[stocks[k]][f],bins=50)
            axs[i][j].grid()
            k+=1
    fig.tight_layout(pad=1.0)
    plt.show()

# Q9 and Q10
for stock in stocks:
    axes=pd.plotting.scatter_matrix(df_dict[stock],hist_kwds={'bins':50},alpha=0.5,s=10,diagonal='kde',figsize=(12,8))
    plt.suptitle("Scatter plot between all features of " + stock + " stock")
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.show()
