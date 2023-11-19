import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pearsoncorrcoef(X,Y,meanX,meanY,varX,varY,numObsv):
    coef_num = 0
    for i in range(np.size(X)):
        coef_num += (X[i] - meanX) * (Y[i] - meanY)
    pcoeff = coef_num / (np.sqrt(varX * (numObsv - 1) * varY * (numObsv - 1)))
    return pcoeff

# Q1. Python program to generate random variables
#     Using the seed as suggested in the assignment
np.random.seed(5764)
# Enter mean, variance and number of observations of each variable in a line with space between each values
# Comment lines 19,20 and uncomment lines 17,18 to give values as part of program and not input
# meanX,varX,numX=0,1,1000
# meanY,varY,numY=5,2,1000
meanX,varX,numX=map(int,input().split())
meanY,varY,numY=map(int,input().split())
X=np.random.normal(meanX,np.sqrt(varX),numX)
Y=np.random.normal(meanY,np.sqrt(varY),numY)

# Q2. Calculated Pearson's correlation coefficient using the below formula
# I am still calculating the mean even though we gave it in the input to get accurate mean

meanX=np.mean(X)
meanY=np.mean(Y)
varX=np.var(X)
varY=np.var(Y)
pcoeff=pearsoncorrcoef(X,Y,meanX,meanY,varX,varY,numX)
#
# # Q3. Printing mean, variance and correlation coefficient for both variables X and Y
print(f"The sample mean of random variable x is :{meanX:.2f}")
print(f"The sample mean of random variable y is :{meanY:.2f}")
print(f"The sample variance of random variable x is :{varX:.2f}")
print(f"The sample variance of random variable y is :{varY:.2f}")
print(f"The sample Pearson’s correlation coefficient between x & y is :{pcoeff:.2f}")
#
# # Q4. Plotting these random variables x and y with xlabel, ylabel, title and legend
plt.plot(X)
plt.plot(Y, color='red')
plt.title("Plot of 2 random normally distributed variables")
plt.xlabel("Observation number")
plt.ylabel("Observation value")
plt.legend(["X","Y"])
plt.show()
#
# # Q5. Plotting an histogram of both x and y with xlabel,ylabel,title,legend
plt.hist(X, 10,density=False)
plt.hist(Y, 10, color="red", density=False)
plt.title("Histogram of 2 random normally distributed variables")
plt.xlabel("Observation number")
plt.ylabel("Observation value")
plt.legend(["X","Y"])
plt.show()

# Q6. Read tute1.csv from github link
df = pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv")
print(df.head().to_string())

# Q7. Calculate pearson correlation coefficient between each pair of columns Sales, AdBudget, GDP
pccSA=pearsoncorrcoef(df['Sales'],df['AdBudget'],np.mean(df['Sales']),np.mean(df['AdBudget']),np.var(df['Sales']),np.var(df['AdBudget']),df.shape[0])
pccSG=pearsoncorrcoef(df['Sales'],df['GDP'],np.mean(df['Sales']),np.mean(df['GDP']),np.var(df['Sales']),np.var(df['GDP']),df.shape[0])
pccAG=pearsoncorrcoef(df['AdBudget'],df['GDP'],np.mean(df['AdBudget']),np.mean(df['GDP']),np.var(df['AdBudget']),np.var(df['GDP']),df.shape[0])

# Q8. Print the correlation coefficients calculated above in the format given in the question
print(f"The sample Pearson’s correlation coefficient between Sales & AdBudget is: {pccSA:.2f}")
print(f"The sample Pearson’s correlation coefficient between Sales & GDP is: {pccSG:.2f}")
print(f"The sample Pearson’s correlation coefficient between AdBudget & GDP is: {pccAG:.2f}")

# Q9. Line plot of Sales, AdBudget and GDP vs time by setting 'Date' as index which gives Date on x-axis
df=df.set_index('Date')
df['Sales'].plot()
df['AdBudget'].plot()
df['GDP'].plot()
plt.title("Plot of Sales, AdBudget,GDP vs Date")
plt.xlabel("Date")
plt.ylabel("US($)")
plt.legend(["Sales","AdBudget","GDP"])
plt.show()

# Q10. Histogram of Sales, AdBudget and GDP vs time
df['Sales'].hist()
df['AdBudget'].hist()
df['GDP'].hist()
plt.title("Histogram of Sales, AdBudget and GDP")
plt.xlabel("US($)")
plt.ylabel("Frequency")
plt.legend(["Sales", "AdBudget","GDP"])
plt.show()
