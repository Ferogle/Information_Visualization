import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Q1.
df=pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv")
df['China_sum']=np.array([0 for i in range(df.shape[0])])
df=df.drop(0)
for i in df.columns.to_list():
    if i.startswith('China.'):
        df['China_sum']=df['China_sum']+df[i].astype('float')
print(df.head().to_string())
#
# Q2.
df['United Kingdom_sum']=np.array([0 for i in range(df.shape[0])])
for i in df.columns.to_list():
    if i.startswith('United Kingdom.'):
        df['United Kingdom_sum'] = df['United Kingdom_sum'] + df[i].astype('float')
print(df.head().to_string())

# Q3.
df['Date']=pd.to_datetime(df['Country/Region'])
plt.plot( df['Date'], df['US'])
plt.title("Confirmed cases of the US vs. time")
plt.xlabel("Date")
plt.ylabel("Confirmed cases")
plt.xticks(rotation=90)
plt.grid()
plt.show()

# Q4.
countries=['United Kingdom_sum','China_sum','Germany','Brazil','India','Italy','US']
for i in countries:
    plt.plot(df['Date'],df[i])
plt.title("Confirmed cases of various countires vs. time")
plt.xlabel("Date")
plt.ylabel("Confirmed cases")
plt.xticks(rotation=90)
plt.grid()
plt.legend()
plt.show()

# Q5.
df.index=df['Date']
print(df.tail().to_string())
df_grp=df[['Date','US']].groupby(pd.Grouper(key='Date', axis=0, freq='6D')).sum()
df_grp=df_grp.reset_index()
print(df_grp.head().to_string())
plot=sns.barplot(data=df_grp, x='Date', y='US')
plt.title("Histogram of confirmed cases of US over time")
date_labels = [pd.to_datetime(label).strftime('%b %Y') for label in df_grp['Date']]
plt.gca().set_xticklabels(date_labels)
plt.xlabel("Date")
plt.ylabel("Confirmed cases")
plt.show()

# Q6.
fig,axes=plt.subplots(3,2)
k=0
for i in range(3):
    for j in range(2):
        df_grp = df[['Date', countries[k]]].groupby(pd.Grouper(key='Date', axis=0, freq='6D')).sum()
        df_grp = df_grp.reset_index()
        plot = sns.barplot(data=df_grp, x='Date', y=countries[k],ax=axes[i,j])
        axes[i,j].set_title(f"Histogram of Confirmed cases of {countries[k]} over time",fontsize=7)
        date_labels = [pd.to_datetime(label).strftime('%Y-%M') for label in df_grp['Date']]
        tick_positions = df_grp.index[::10]
        tick_labels = df_grp['Date'].dt.strftime('%b %Y').iloc[::10]
        axes[i,j].set_xticks(tick_positions)
        axes[i,j].set_xticklabels(tick_labels,rotation=45,fontsize=7)
        k+=1
plt.tight_layout()
plt.show()

# Q7.
print(df[countries].describe().to_string())

# ===================================================================================
# Second part of Assignment

# Q1.
df=sns.load_dataset('titanic')
print(df.head().to_string())
orig_shape=df.shape
df.dropna(inplace=True)
cl_shape=df.shape
print(df.head().to_string())
print(f"Percentage of data imputed: {round((cl_shape[0]*100)/orig_shape[0],2)}")

# Q2.
def disp_counts(vals):
    def my_autopct(pct):
        return '{v:.0f}'.format(v=(sum(vals)*pct)/100)
    return my_autopct
df_count=df['sex'].value_counts()
plt.pie(df_count, autopct=disp_counts(df_count.values),labels=df_count.index)
plt.title("Pie chart of total people of Titanic")
plt.legend()
plt.show()
#
# # Q3.
plt.pie(df_count, labels=df_count.index, autopct="%1.1f%%")
plt.title("Pie chart of total people of Titanic in %")
plt.legend()
plt.show()
#
# # Q4.
df_male_sur=df[df['sex']=='male']
df_ms_val=df_male_sur['survived'].value_counts()
plt.pie(df_ms_val[::-1],labels=['Male survived','Male not survived'],autopct='%1.2f%%')
plt.title('Pie chart of Male survival in Titanic')
plt.legend()
plt.show()
#
# # Q5.
df_female_sur=df[df['sex']=='female']
df_fms_val=df_female_sur['survived'].value_counts()
plt.pie(df_fms_val,labels=['Female survived','Female not survived'],autopct='%1.2f%%')
plt.title('Pie chart of Female survival in Titanic')
plt.legend()
plt.show()

# Q6.
df_pclass=df['pclass'].value_counts()
plt.pie(df_pclass,labels=['ticket class 1','ticket class 2','ticket class 3'],autopct='%1.1f%%')
plt.title('Pie chart of passengers based on ticket level in Titanic')
plt.legend()
plt.show()
#
# # Q7.
df_pclass_sur=df[df['survived']==1]['pclass'].value_counts()
plt.pie(df_pclass_sur,labels=['ticket class 1','ticket class 2','ticket class 3'],autopct='%1.1f%%')
plt.title('Pie chart of passengers survived based on ticket level in Titanic')
plt.legend()
plt.show()

# Q8.
df_pclass_sur=df[df['pclass']==1]['survived'].value_counts()
plt.pie(df_pclass_sur,labels=['Survival rate','Death rate'],autopct='%1.1f%%')
plt.title('Survival rate & Death Rate: Ticket class 1')
plt.legend()
plt.show()
#
# # Q9.
df_pclass_sur=df[df['pclass']==2]['survived'].value_counts()
plt.pie(df_pclass_sur,labels=['Survival rate','Death rate'],autopct='%1.1f%%')
plt.title('Survival rate & Death Rate: Ticket class 2')
plt.legend()
plt.show()

# Q10.
df_pclass_sur=df[df['pclass']==3]['survived'].value_counts()
plt.pie(df_pclass_sur,labels=['Survival rate','Death rate'],autopct='%1.1f%%')
plt.title('Survival rate & Death Rate: Ticket class 3')
plt.legend()
plt.show()
#
# # Q11.
plt.figure(figsize=(16,8))
fig, axes = plt.subplots(3, 3)
df_count=df['sex'].value_counts()
axes[0,0].pie(df_count, autopct=disp_counts(df_count.values),labels=df_count.index,textprops={'fontsize':5})
axes[0,0].set_title("Pie chart of total people of Titanic",fontsize=5)
axes[0,0].legend(loc='upper right', fontsize=5,bbox_to_anchor=(1.4,1.0))

axes[0,1].pie(df_count, labels=df_count.index, autopct="%1.1f%%",textprops={'fontsize':5})
axes[0,1].set_title("Pie chart of total people of Titanic in %",fontsize=5)
axes[0,1].legend(loc='upper right', fontsize=5,bbox_to_anchor=(1.4,1.0))

df_male_sur=df[df['sex']=='male']
df_ms_val=df_male_sur['survived'].value_counts()
axes[0,2].pie(df_ms_val[::-1],labels=['Male survived','Male not survived'],autopct='%1.2f%%',textprops={'fontsize':5})
axes[0,2].set_title('Pie chart of Male survival in Titanic',fontsize=5)
axes[0,2].legend(loc='upper right', fontsize=5,bbox_to_anchor=(1.9,0.8))

df_female_sur=df[df['sex']=='female']
df_fms_val=df_male_sur['survived'].value_counts()
axes[1,0].pie(df_fms_val[::-1],labels=['Female not survived','Female survived'],autopct='%1.2f%%',textprops={'fontsize':5})
axes[1,0].set_title('Pie chart of Female survival in Titanic',fontsize=5)
axes[1,0].legend(loc='upper right', fontsize=5,bbox_to_anchor=(1.9,0.8))

df_pclass=df['pclass'].value_counts()
axes[1,1].pie(df_pclass,labels=['class 1','class 2','class 3'],autopct='%1.1f%%',textprops={'fontsize':5})
axes[1,1].set_title('Pie chart of passengers based on ticket level in Titanic',fontsize=5)
axes[1,1].legend(loc='upper right', fontsize=5,bbox_to_anchor=(1.4,1.0))

df_pclass_sur=df[df['survived']==1]['pclass'].value_counts()
axes[1,2].pie(df_pclass_sur,labels=['class 1','class 2','class 3'],autopct='%1.1f%%',textprops={'fontsize':5})
axes[1,2].set_title('Pie chart of survived based on ticket level in Titanic',fontsize=5)
axes[1,2].legend(loc='upper right', fontsize=5,bbox_to_anchor=(1.4,1.0))

df_pclass_sur=df[df['pclass']==1]['survived'].value_counts()
axes[2,0].pie(df_pclass_sur,labels=['Survival rate','Death rate'],autopct='%1.1f%%',textprops={'fontsize':5})
axes[2,0].set_title('Survival rate & Death rate: Ticket class 1',fontsize=5)
axes[2,0].legend(loc='upper right', fontsize=5,bbox_to_anchor=(1.4,1.0))

df_pclass_sur=df[df['pclass']==2]['survived'].value_counts()
axes[2,1].pie(df_pclass_sur,labels=['Survival rate','Death rate'],autopct='%1.1f%%',textprops={'fontsize':5})
axes[2,1].set_title('Survival rate & Death Rate: Ticket class 2',fontsize=5)
axes[2,1].legend(loc='upper right', fontsize=5,bbox_to_anchor=(1.4,1.0))

df_pclass_sur=df[df['pclass']==3]['survived'].value_counts()
axes[2,2].pie(df_pclass_sur,labels=['Survival rate','Death rate'],autopct='%1.1f%%',textprops={'fontsize':5})
axes[2,2].set_title('Survival rate & Death Rate: Ticket class 3',fontsize=5)
axes[2,2].legend(loc='upper right', fontsize=5,bbox_to_anchor=(1.9,0.8))

plt.subplots_adjust(wspace=0.6, hspace=0.6)
plt.show()