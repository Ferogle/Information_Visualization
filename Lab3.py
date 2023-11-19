import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from prettytable import PrettyTable
from matplotlib.gridspec import GridSpec
sns.set_style("whitegrid")

# Q1.
df=pd.read_excel("Sample - Superstore.xls")
df=df.drop(['Row ID', 'Order ID', 'Customer ID','Customer Name', 'Postal Code',
'Product ID','Order Date', 'Ship Date', 'Country', 'Segment'], axis=1)
print(df.head().to_string())

# Q2.
df_agg = df.groupby('Category').sum().reset_index()
print(df_agg)
fig,ax=plt.subplots(2,2, figsize=(18,18))
ax[0,0].pie(x=df_agg['Sales'],labels=df_agg['Category'].unique(),autopct="%1.2f%%",explode=[0,0.4,0],textprops={'fontsize':30})
ax[0,0].set_title("Total sales of each category",fontfamily='serif',fontsize=35,color='blue')
ax[0,1].pie(x=df_agg['Quantity'],labels=df_agg['Category'].unique(),autopct="%1.2f%%",explode=[0,0,0.4],textprops={'fontsize':30})
ax[0,1].set_title("Total units sold of each category",fontfamily='serif',fontsize=35,color='blue')
ax[1,0].pie(x=df_agg['Discount'],labels=df_agg['Category'].unique(),autopct="%1.2f%%",explode=[0,0,0.4],textprops={'fontsize':30})
ax[1,0].set_title("Total discount of each category",fontfamily='serif',fontsize=35,color='blue')
ax[1,1].pie(x=df_agg['Profit'],labels=df_agg['Category'].unique(),autopct="%1.2f%%",explode=[0.4,0,0],textprops={'fontsize':30})
ax[1,1].set_title("Total profit of each category",fontfamily='serif',fontsize=35,color='blue')
plt.show()
print("In total sales plot (ax[0,0]), the maximum category is Technology and minimum category is Office supplies")
print("In total quantity plot (ax[0,1]), the maximum category is Office supplies and minimum category is Technology")
print("In total discount plot (ax[1,0]), the maximum category is Office supplies and minimum category is Technology")
print("In total profit plot (ax[1,1]), the maximum category is Technology and minimum category is Furniture")

# Q3.
pt=PrettyTable()
pt.title="Super store - Category"
pt.field_names=['','Sales($)','Quantity','Discount($)','Profit($)']
for i in range(len(df_agg)):
    row=[df_agg.loc[i,'Category'],round(df_agg.loc[i,'Sales'],2),round(df_agg.loc[i,'Quantity'],2),round(df_agg.loc[i,'Discount'],2),round(df_agg.loc[i,'Profit'],2)]
    pt.add_row(row)
row=['Maximum value',round(max(df_agg['Sales']),2),round(max(df_agg['Quantity']),2),round(max(df_agg['Discount']),2),round(max(df_agg['Profit']),2)]
pt.add_row(row)
row=['Minimum value',round(min(df_agg['Sales']),2),round(min(df_agg['Quantity']),2),round(min(df_agg['Discount']),2),round(min(df_agg['Profit']),2)]
pt.add_row(row)
row=['Maximum feature','Technology','Office supplies','Office supplies','Technology']
pt.add_row(row)
row=['Minimum feature','Office supplies','Furniture','Technology','Furniture']
pt.add_row(row)
print(pt)

# Q4.
df_agg1=df.groupby('Sub-Category').sum().reset_index()
df_agg1=df_agg1.sort_values(by='Sales',ascending=False).iloc[:10,:]
f,ax=plt.subplots(figsize=(20,8))
ax.bar(x=df_agg1['Sub-Category'],height=df_agg1['Sales'],width=0.4,edgecolor='blue',color='#95DEE3',label='Sales')
ax.plot(df_agg1['Sub-Category'],df_agg1['Profit'],color='red',linewidth=4,marker='o',label='Profit')
plt.title("Profit and Sales per sub-category",fontsize=30)
for i,j in zip(df_agg1['Sub-Category'],df_agg1['Sales']):
    if j>250000:
        ax.text(i,j-80000,'$'+str(round(j,2)),ha='center',va='bottom',rotation=90,fontsize=20)
    else:
        ax.text(i,j,'$'+str(round(j, 2)),ha='center',va='bottom',rotation=90,fontsize=20)
ax2 = ax.twinx()
ax.tick_params(labeltop=False,labelright=True,labelsize=20)
ax2.set_ylabel("USD($)",fontsize=25,labelpad=70)
ax.set_xlabel("Sub_category",fontsize=25)
ax.set_ylabel("USD($)",fontsize=25)
yticks=list(np.arange(-50000,400000,50000))
ax2.tick_params(labelsize=20)
ax.set_yticks(yticks)
ax2.set_yticks([])
plt.grid()
ax.legend(bbox_to_anchor=(1.08,1.1),loc='upper right')
plt.tight_layout()
plt.show()

# Q5.
x = np.linspace(0, 2 * np.pi, 100)  # 100 points between 0 and 2π
y = np.sin(x)
y_cos = np.cos(x)
plt.plot(x,y,'b--',label='sine wave',linewidth=3)
plt.plot(x,y_cos,'r-.',label='cosine wave',linewidth=3)
plt.fill_between(x,y,y_cos,where=(y>y_cos),interpolate=True,color='green',alpha=0.3)
plt.fill_between(x,y,y_cos,where=(y<y_cos),interpolate=True,color='orange',alpha=0.3)
point_to_annotate = (2, 0.25)
annotation_text = 'area where sine is greater than cosine'
plt.annotate(annotation_text, xy=point_to_annotate, xytext=(3, 1),
             arrowprops=dict(arrowstyle='->',color='green',ls='dashed'),
             fontsize=10,fontfamily='serif',weight='bold')
plt.xlabel('x-axis',fontfamily='serif',fontsize=15,color='darkred')
plt.ylabel('y-axis',fontfamily='serif',fontsize=15,color='darkred')
plt.title('Fill between x-axis and plot line',fontfamily='serif',fontsize=20,color='blue')
plt.grid(True)
plt.legend(loc='lower left',prop=FontProperties(size=15))
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.show()

# Q6.
x = np.linspace(-4, 4, 800)
y = np.linspace(-4, 4, 800)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X*X + Y*Y))
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm',alpha=1,linewidth=0)
ax.contour(X, Y, Z, zdir='z', offset=-6, cmap='coolwarm',linewidths=1)
ax.contour(X, Y, Z, zdir='x', offset=-5, cmap='coolwarm',linewidths=1)
ax.contour(X, Y, Z, zdir='y', offset= 5, cmap='coolwarm',linewidths=1)
ax.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(-6, 2), xlabel='X', ylabel='Y', zlabel='Z')
ax.set_xlabel('X Label',fontfamily='serif',fontsize=15,color='darkred')
ax.set_ylabel('Y Label',fontfamily='serif',fontsize=15,color='darkred')
ax.set_zlabel('Z Label',fontfamily='serif',fontsize=15,color='darkred')
# ax.set_zticks(list(np.arange(-6,3,1)))
ax.set_yticks(list(np.arange(-5,6,1)))
ax.set_xticks(list(np.arange(-5,6,1)))
ax.set_title(r'Surface plot of z = sin$\sqrt{x^2+y^2}$',fontfamily='serif',fontsize=25,color='blue')
# plt.tight_layout()
plt.show()

# Q7.
f=plt.figure(figsize=(9,7))
gs=GridSpec(2,2)
print(df_agg1.head().to_string())
x=np.arange(len(df_agg1['Sub-Category']))
ax0=plt.subplot(gs[0,:])
ax0.bar(x-0.2,df_agg1['Sales'],0.4,edgecolor='blue',color='#95DEE3',label='Sales')
ax0.bar(x+0.2,df_agg1['Profit'],0.4,edgecolor='red',color='lightcoral',label='Profit')
ax0.set_title("Sales and Profit per sub-category",fontsize=15)
ax0.set_xlabel("Sub-Category")
ax0.set_ylabel("USD($)")
ax0.set_yticks(list(np.arange(-50000,400000,50000)))
ax0.set_xticks(list(x))
ax0.set_xticklabels(list(df_agg1['Sub-Category'].unique()))
ax0.tick_params(axis='x', labelsize=10)  # Set the size of x-axis tick labels
ax0.tick_params(axis='y', labelsize=10)
ax0.legend()

ax2=plt.subplot(gs[1,0])
ax2.pie(x=df_agg['Sales'],labels=df_agg['Category'].unique(),autopct="%1.2f%%")
ax2.set_title('Sales',fontsize=15)
ax2=plt.subplot(gs[1,1])
ax2.pie(x=df_agg['Profit'],labels=df_agg['Category'].unique(),autopct="%1.2f%%")
ax2.set_title('Profit',fontsize=15)

plt.tight_layout()
plt.show()