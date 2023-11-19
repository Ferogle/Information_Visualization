

import numpy as np
from pandas_datareader import data
from prettytable import PrettyTable
import yfinance as yf
yf.pdr_override()
dfs=[]
features=['High','Low','Open','Close','Volume','Adj Close']
stocks = ['AAPL','ORCL','TSLA','IBM','YELP','MSFT']
nstocks = len(stocks)
nfeatures = len(features)
# Q1: Read stock data from yahoo
for comp in stocks:
    dfs.append(data.get_data_yahoo(comp,start='2013-01-01',end='2023-08-28'))

# Q2: Mean value comparison using a PrettyTable()
pt_mean=PrettyTable()
pt_mean.title = "Mean value comparison"
pt_mean.field_names=["Name\\Feature"]+features
print(pt_mean.field_names)
row=[]
for i in range(len(stocks)):
    row.append(stocks[i])
    for j in features:
        row.append(round(np.mean(dfs[i][j]),2))
    print(row)
    pt_mean.add_row(row)
    row=[]
mxrow=["Maximum Value"]
for feature in range(1,nfeatures+1):
    mxrow.append(max(pt_mean.rows[i][feature] for i in range(nstocks)))
pt_mean.add_row(mxrow)
mnrow=["Minimum Value"]
for feature in range(1,nfeatures+1):
    mnrow.append(min(pt_mean.rows[i][feature] for i in range(nstocks)))
pt_mean.add_row(mnrow)

mxcomp=["Maximum Company Name"]
for feature in range(1,nfeatures+1):
    for comp in range(nstocks):
        if pt_mean.rows[nstocks][feature]==pt_mean.rows[comp][feature]:
            mxcomp.append(stocks[comp])
            break
pt_mean.add_row(mxcomp)
mncomp=["Minimum Company Name"]
for feature in range(1,nfeatures+1):
    for comp in range(nstocks):
        if pt_mean.rows[nstocks+1][feature]==pt_mean.rows[comp][feature]:
            mncomp.append(stocks[comp])
            break
pt_mean.add_row(mncomp)
print(pt_mean)

# Q3: Variance comparison table
pt_var = PrettyTable()
pt_var.title = "Variance comparison"
pt_var.field_names=["Name\Feature"]+features
row=[]
for i in range(len(stocks)):
    row.append(stocks[i])
    for j in features:
        row.append(round(np.mean(dfs[i][j]),2))
    print(row)
    pt_var.add_row(row)
    row=[]
mxrow=["Maximum Value"]
for feature in range(1,nfeatures+1):
    mxrow.append(max(pt_var.rows[i][feature] for i in range(nstocks)))
pt_var.add_row(mxrow)
mnrow=["Minimum Value"]
for feature in range(1,nfeatures+1):
    mnrow.append(min(pt_var.rows[i][feature] for i in range(nstocks)))
pt_var.add_row(mnrow)

mxcomp=["Maximum Company Name"]
for feature in range(1,nfeatures+1):
    for comp in range(nstocks):
        if pt_var.rows[nstocks][feature]==pt_var.rows[comp][feature]:
            mxcomp.append(stocks[comp])
            break
pt_var.add_row(mxcomp)
mncomp=["Minimum Company Name"]
for feature in range(1,nfeatures+1):
    for comp in range(nstocks):
        if pt_var.rows[nstocks+1][feature]==pt_var.rows[comp][feature]:
            mncomp.append(stocks[comp])
            break
pt_var.add_row(mncomp)
print(pt_var)

# Q4. Standard Deviation value comparison table
pt_std=PrettyTable()
pt_std.title = "Standard Deviation Value comparison"
pt_std.field_names=["Name\Feature"]+features
row=[]
for i in range(len(stocks)):
    row.append(stocks[i])
    for j in features:
        row.append(round(np.mean(dfs[i][j]),2))
    print(row)
    pt_std.add_row(row)
    row=[]
mxrow=["Maximum Value"]
for feature in range(1,nfeatures+1):
    mxrow.append(max(pt_std.rows[i][feature] for i in range(nstocks)))
pt_std.add_row(mxrow)
mnrow=["Minimum Value"]
for feature in range(1,nfeatures+1):
    mnrow.append(min(pt_std.rows[i][feature] for i in range(nstocks)))
pt_std.add_row(mnrow)

mxcomp=["Maximum Company Name"]
for feature in range(1,nfeatures+1):
    for comp in range(nstocks):
        if pt_std.rows[nstocks][feature]==pt_std.rows[comp][feature]:
            mxcomp.append(stocks[comp])
            break
pt_std.add_row(mxcomp)
mncomp=["Minimum Company Name"]
for feature in range(1,nfeatures+1):
    for comp in range(nstocks):
        if pt_std.rows[nstocks+1][feature]==pt_std.rows[comp][feature]:
            mncomp.append(stocks[comp])
            break
pt_std.add_row(mncomp)
print(pt_std)

# Q5. Median value comparison table
pt_median = PrettyTable()
pt_median.title = "Median Value comparison"
pt_median.field_names=["Name\Feature"]+features
row=[]
for i in range(len(stocks)):
    row.append(stocks[i])
    for j in features:
        row.append(round(np.mean(dfs[i][j]),2))
    print(row)
    pt_median.add_row(row)
    row=[]
mxrow=["Maximum Value"]
for feature in range(1,nfeatures+1):
    mxrow.append(max(pt_median.rows[i][feature] for i in range(nstocks)))
pt_median.add_row(mxrow)
mnrow=["Minimum Value"]
for feature in range(1,nfeatures+1):
    mnrow.append(min(pt_median.rows[i][feature] for i in range(nstocks)))
pt_median.add_row(mnrow)
mxcomp=["Maximum Company Name"]
for feature in range(1,nfeatures+1):
    for comp in range(nstocks):
        if pt_median.rows[nstocks][feature]==pt_median.rows[comp][feature]:
            mxcomp.append(stocks[comp])
            break
pt_median.add_row(mxcomp)
mncomp=["Minimum Company Name"]
for feature in range(1,nfeatures+1):
    for comp in range(nstocks):
        if pt_median.rows[nstocks+1][feature]==pt_median.rows[comp][feature]:
            mncomp.append(stocks[comp])
            break
pt_median.add_row(mncomp)
print(pt_median)

# Q6, Q7 Printing correlation matrices of all features of each stock
# My stocks array first contains 'AAPL' so its printed in the first place and later stocks follow
for i in range(nstocks):
    print(f"\n\n {stocks[i]} stock")
    print(np.round_(dfs[i].corr(),2))
         
# Q8. Table to support my answer in the report
all_std = PrettyTable()
all_std.title="STD of Adj Close of all stocks"
all_std.field_names = ["Name", "Adj Close Std"]
for i in range(nstocks):
    all_std.add_row([stocks[i],round(np.mean(dfs[i]['Adj Close']),2)])
print(all_std)