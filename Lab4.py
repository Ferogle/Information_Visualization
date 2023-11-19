import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Q1.
df=px.data.stocks()
print(df.columns)
print(df.head())

# Q2.
fig=px.line(data_frame=df,x='date',y=['GOOG','AAPL','AMZN','FB','NFLX','MSFT'],
            height=800,
            width=2000,
            template='plotly_dark',
            title='Stock values - Major Tech company')
fig.update_layout(
title_font_family = 'Times New Roman',
title_font_size = 30,
title_font_color = 'red',
legend_title_font_color = 'green',
legend_title_font_size = 20,
font_family = 'Courier New',
font_size = 30,
font_color = 'yellow',
title={'y': .95, 'x': 0.5},
xaxis_title='Time',
yaxis_title='Normalised ($)'
)
fig.update_traces(
line = dict(width = 4)
)
fig.show(renderer = 'browser')

# Q3.
companies = ['GOOG','AAPL','AMZN','FB','NFLX','MSFT']
fig = make_subplots(rows=3, cols=2)
k=0
for i in range(3):
    for j in range(2):
        hist_plot = go.Histogram(x=df[companies[k]], nbinsx=50, name=companies[k])
        fig.add_trace(hist_plot,row=i+1,col=j+1)
        k+=1

for i in range(3):
    for j in range(2):
        fig.update_xaxes(title_text='Normalised Price ($)',row=i+1,col=j+1)
        fig.update_yaxes(title_text='Frequency',row=i+1,col=j+1)

fig.update_layout(
    title_text='Histogram Plot',
    title_font_family = 'Times New Roman',
    title_font_size = 30,
    title_font_color = 'red',
    legend_title_font_color = 'green',
    legend_title_font_size = 30,
    font_family = 'Courier New',
    font_size = 15,
    font_color = 'black',
    title={'y': .95, 'x': 0.5},
    showlegend=True
)

fig.show(renderer = 'browser')

# Q4.a
df['date']=pd.to_datetime(df['date'])
df.set_index('date',inplace=True)
std=StandardScaler()
for i in companies:
    df[i]=std.fit_transform(df[i].values.reshape(-1,1))
# print(df.head())
_, sing, _ = np.linalg.svd(df)
print("Singular values")
print(sing.round(2))
print(f"Condition number: {round(np.max(sing)/np.min(sing),2)}")

corr_mat = df.corr()
sns.heatmap(data=corr_mat,annot=True,fmt="0.2f")
plt.title("Correlation coefficient between features-Original feature space")
plt.show()

pca=PCA()
pca.fit(df)
X_trnf = pca.transform(df)
df_trnf = pd.DataFrame(X_trnf, columns=[f'Principal col {i}' for i in range(1, 7)])
print(f"Components {pca.components_}")
print(f"Explained variance ratio {100*pca.explained_variance_ratio_.round(2)}")
cum_exp_var_orig = np.cumsum(100*pca.explained_variance_ratio_).round(2)
print(f"Cumulative Explained variance ratio {cum_exp_var_orig}")

pca_reduced = PCA(n_components=4)
X_reduced = pca_reduced.fit_transform(df)
df_reduced = pd.DataFrame(X_reduced, columns=[f'Principal col {i}' for i in range(1, 5)])
print(f"Explained variance ratio of reduced feature space{100*pca_reduced.explained_variance_ratio_.round(2)}")
cum_exp_var_red = np.cumsum(100*pca_reduced.explained_variance_ratio_).round(2)
print(f"Cumulative Explained variance ratio of reduced feature space{cum_exp_var_red}")

plt.plot(np.arange(1,len(cum_exp_var_orig)+1), cum_exp_var_orig)
x_point=4
y_point=95
plt.axvline(x=x_point, linestyle='--', label=f'x = {x_point}')
plt.axhline(y=y_point, linestyle='--', label=f'y = {y_point}')
plt.title("Cumulative explained variance vs. number of components")
plt.xlabel('# of components')
plt.ylabel('Cumulative explained variance')
plt.show()

_, sing, _ = np.linalg.svd(df_reduced)
print("Singular values of reduced feature space")
print(sing.round(2))
print(f"Condition number of reduced feature space: {round(np.max(sing)/np.min(sing),2)}")

corr_mat = df_reduced.corr()
sns.heatmap(data=corr_mat,annot=True,fmt="0.2f")
plt.title("Correlation coefficient between features-Reduced feature space")
plt.show()

# generalised function for number of components
def tranform_comp(df, n):
    pca_n = PCA(n_components=n)
    X_transformed = pca_n.fit_transform(df)
    df_transformed = pd.DataFrame(X_transformed, columns=[f'Principal col {i}' for i in range(1, n+1)])
    return df_transformed

print("Reduced components (4) with transformed data")
print(df_reduced.head())
print("Original components (6) with tranformed data")
print(df_trnf.head().to_string())
print("Tranformed data with 5 components")
print(tranform_comp(df,5).head().to_string())

# Q4.i
df_reduced=df_reduced.set_index(df.index)
# df_reduced.reset_index()
fig=px.line(data_frame=df_reduced,y=['Principal col 1','Principal col 2','Principal col 3','Principal col 4'],
            height=800,
            width=2000,
            template='plotly_dark',
            title='Stock values standardised - Principal component feature space')
fig.update_layout(
title_font_family = 'Times New Roman',
title_font_size = 30,
title_font_color = 'red',
legend_title_font_color = 'green',
legend_title_font_size = 20,
font_family = 'Courier New',
font_size = 30,
font_color = 'yellow',
title={'y': .95, 'x': 0.5},
xaxis_title='Time',
yaxis_title='Normalised ($)'
)
fig.update_traces(
line = dict(width = 4)
)
fig.show(renderer = 'browser')

# Q4.j
fig = make_subplots(rows=4, cols=1)
companies=['Principal col 1','Principal col 2','Principal col 3','Principal col 4']
k=0
for i in range(4):
    hist_plot = go.Histogram(x=df_reduced[companies[k]], nbinsx=50, name=companies[k])
    fig.add_trace(hist_plot,row=i+1,col=1)
    k+=1

for i in range(4):
    fig.update_xaxes(title_text='Normalised Price ($)',row=i+1,col=1)
    fig.update_yaxes(title_text='Frequency',row=i+1,col=1)

fig.update_layout(
    title_text='Histogram Plot',
    title_font_family = 'Times New Roman',
    title_font_size = 30,
    title_font_color = 'red',
    legend_title_font_color = 'green',
    legend_title_font_size = 30,
    font_family = 'Courier New',
    font_size = 15,
    font_color = 'black',
    title={'y': .95, 'x': 0.5},
    showlegend=True
)

fig.show(renderer = 'browser')

# Q4. k
fig=px.scatter_matrix(df)
fig.update_layout(
    title_text='Scatter plot of original feature space'
)
fig.update_traces(diagonal_visible=False)
fig.show(renderer = 'browser')

fig=px.scatter_matrix(df_reduced)
fig.update_layout(
    title_text='Scatter plot of reduced feature space'
)
fig.update_traces(diagonal_visible=False)
fig.show(renderer = 'browser')
