import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

df=pd.read_csv('Airlines.csv')
print(df.head().to_string())
print(df.dtypes)

missing_values = df.isna().sum().sum() + df.isnull().sum().sum()
print(f"Missing values in the dataset: {missing_values}")

encode_features = ['Airline', 'AirportFrom', 'AirportTo']
le=LabelEncoder()
for i in encode_features:
    df[i] = le.fit_transform(df[i])
print(df.head().to_string())

numerical_features = ['Length','Time']
categorical_features = ['Airline','Flight','AirportFrom','AirportTo','DayOfWeek']
target = 'Delay'

for i in categorical_features:
    print(df[i].value_counts()[:20])

# fig,axs=plt.subplots(2,2, figsize=(12,8))
# axs[0][0].plot(df[df['Flight']==16]['Length'],color='r')
# axs[0][0].set_title('Length(s) of Flight 16')
# axs[0][0].set_xlabel('# of observations')
# axs[0][1].plot(df[df['Flight']==5]['Length'])
# axs[0][1].set_title('Length(s) of Flight 5')
# axs[0][1].set_xlabel('# of observations')
# axs[0][1].set_ylabel('Length (in mins)')
# axs[1][0].plot(df[df['Flight']==9]['Length'])
# axs[1][0].set_title('Length(s) of Flight 9')
# axs[1][0].set_xlabel('# of observations')
# axs[1][0].set_ylabel('Length (in mins)')
# axs[1][1].plot(df[df['Flight']==8]['Length'])
# axs[1][1].set_title('Length(s) of Flight 8')
# axs[1][1].set_xlabel('# of observations')
# axs[1][1].set_ylabel('Length (in mins)')
# fig.suptitle("Plot of lengths of top 4 most commonly used type of flights")
# plt.tight_layout()
# plt.show()

# df_flight_16 = df[df['Flight']==16]
# df_flight_5 = df[df['Flight']==5]
# df_flight_9 = df[df['Flight']==9]
# df_agg_5 = df_flight_5[['DayOfWeek', 'id']].groupby('DayOfWeek').count().reset_index()
# df_agg_16 = df_flight_16[['DayOfWeek', 'id']].groupby('DayOfWeek').count().reset_index()
# df_agg_9 = df_flight_9[['DayOfWeek', 'id']].groupby('DayOfWeek').count().reset_index()
# plt.bar(df_agg_16['DayOfWeek'], height=df_agg_16['id'],label='Flight 16')
# plt.bar(df_agg_5['DayOfWeek'], height=df_agg_5['id'], bottom=df_agg_16['id'], label='Flight 5')
# plt.bar(df_agg_9['DayOfWeek'], height=df_agg_9['id'], bottom=(df_agg_16['id']+df_agg_5['id']), label='Flight 9')
# plt.xlabel('Day of week')
# plt.ylabel('# of observations')
# plt.title('Stacked bar plot of # of observations per day of week for Flight 5, 16 and 9')
# plt.legend()
# plt.show()

# index=np.arange(7)
# df_airportfrom_atl = df[df['AirportFrom']=='ATL']
# df_airportfrom_ord = df[df['AirportFrom']=='ORD']
# df_airportfrom_dfw = df[df['AirportFrom']=='DFW']
# df_agg_atl = df_airportfrom_atl[['DayOfWeek', 'id']].groupby('DayOfWeek').count().reset_index()
# df_agg_ord = df_airportfrom_ord[['DayOfWeek', 'id']].groupby('DayOfWeek').count().reset_index()
# df_agg_dfw = df_airportfrom_dfw[['DayOfWeek', 'id']].groupby('DayOfWeek').count().reset_index()
# plt.bar(index, df_agg_atl['id'], 0.25, label='AirportFrom ATL')
# plt.bar(index+0.25, df_agg_ord['id'], 0.25, label='AirportFrom ORD')
# plt.bar(index+0.5, df_agg_dfw['id'], 0.25, label='AirportFrom DFW')
# plt.xticks(index+0.25,df_agg_atl['DayOfWeek'])
# plt.xlabel('Day of week')
# plt.ylabel('# of observations')
# plt.title('Group bar plot of # of obs. vs day of week for top 3 AirportFrom values')
# plt.legend()
# plt.tight_layout()
# plt.show()

# print(df['Delay'].value_counts())
# sns.set_style('whitegrid')
# sns.countplot(data=df,x='Delay')
# plt.title('Countplot of # of samples vs Delay')
# plt.xlabel("Whether flight delayed")
# plt.ylabel("# of samples")
# plt.show()

# df_top5_airline = df['Airline'].value_counts()[:5]
# plt.pie(df_top5_airline,labels=df_top5_airline.keys(),autopct="%1.2f%%")
# plt.title('Top 5 Airlines in decreasing order of observations')
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,8))
# sns.displot(df['Airline'],kde=True)
# plt.title('Displot of top 10 airlines')
# plt.xlabel('Airline')
# plt.ylabel('# of samples')
# plt.tight_layout()
# plt.show()

# sns.pairplot(df)
# plt.title('# of samples for different days of week')
# plt.xlabel('Day of week')
# plt.ylabel('# of samples')
# plt.show()

# correlation = df.corr().round(2)
# plt.figure(figsize = (14,7))
# sns.heatmap(correlation, annot = True, cbar=True)
# plt.title('Heatmap of correlation between all the features')
# plt.show()

# plt.figure(figsize=(10,8))
# df_airportto = df[['AirportTo','id']].groupby('AirportTo').count().reset_index()
# df_airportto = df_airportto.sort_values('id', ascending=False)
# sns.histplot(df.head(1000)['AirportTo'], kde=True)
# plt.title('Histogram of 1000 samples of AirportTo')
# plt.xlabel('AirportTo')
# plt.ylabel('# of samples')
# plt.tight_layout()
# plt.show()

# sm.qqplot(df['Time'])
# plt.title('QQ plot of Time feature')
# plt.show()

# df_airline_16_5_9 = df[(df['Flight'] == 16) | (df['Flight'] == 5)]
# print(df_airline_16_5_9.head())
# sns.kdeplot(data=df_airline_16_5_9, x='Length', alpha=0.7, fill=True, hue='Flight', palette='OrRd')
# plt.title('KDE plot of lengths of Flight 16 and flight 5')
# plt.show()

# sns.regplot(x=df['Time'], y=df['Length'], color='blue', scatter_kws={'color': 'red'})
# plt.title("Reg plot between Length and Time")
# plt.xlabel('Time')
# plt.ylabel('Length')
# plt.show()

# sns.set_style('whitegrid')
# sns.boxenplot(x=df['Time'])
# plt.title('Boxen plot of Time feature')
# plt.show()

# plt.stackplot(np.arange(len(df[df['AirportFrom']=='ATL'])),df[df['AirportFrom']=='ATL']['Time'])
# plt.xlabel('observation')
# plt.ylabel('Time')
# plt.title('Area plot of Time for all observations of Airport ATL')
# plt.show()

# sns.violinplot(data=df, x='DayOfWeek', y='Time')
# plt.title('Violin plot of Time for each day of week')
# plt.show()

# sns.jointplot(data=df[df['Airline']=='US'], x='Time', y='Length', hue='DayOfWeek', kind='kde')
# # plt.title('Jointplot of Time vs. Length for each day of week')
# # plt.legend()
# plt.show()

# sns.scatterplot(data=df, x='Time',y='Length')
# sns.rugplot(data=df, x='Time',y='Length')
# plt.title('Rug plot of Length vs. Time with scatter plot')
# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# df_flight16 = df[df['Flight']==16]
# X, Y=np.meshgrid(df_flight16['AirportFrom'], df_flight16['AirportTo'])
# Z = df_flight16['Time'].values.reshape(len(df_flight16['AirportFrom']), len(df_flight16['AirportTo']))
# ax.contour3D(X, Y, Z, cmap='binary')
# plt.show()

sns.clustermap(df[['AirportTo', 'AirportFrom', 'Time', 'Length']].head(500))
plt.title('Cluster map of AirportTo, AirportFrom, Time, Length')
plt.tight_layout()
plt.show()

# plt.hexbin(x=df['AirportFrom'],y=df['AirportTo'],gridsize=10)
# plt.title('Hexbin plot of AiportFrom and AirportTo')
# plt.xlabel('AirportFrom')
# plt.ylabel('AirportTo')
# plt.tight_layout()
# plt.show()

# sns.stripplot(data=df[df['Flight']==16] , x='Length', y='DayOfWeek')
# plt.title('Stripplot between DayOfWeek and Length of Flight 16')
# plt.xlabel('Length')
# plt.ylabel('DayOfWeek')
# # plt.xticks(range(55,300,40),labels=[str(i) for i in range(55,300,40)])
# plt.tick_params(labelrotation=45)
# plt.show()

# df_top4_flight = df[(df['Flight']==16) | (df['Flight']==5) | (df['Flight']==9) | (df['Flight']==8)]
# sns.swarmplot(data=df_top4_flight, x='Flight',y='Time',palette='Set2')
# plt.title('Swarmplot of Flight vs Time for flights 16,5,8,9')
# plt.show()
