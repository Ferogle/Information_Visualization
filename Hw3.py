import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('darkgrid')
pd.options.display.float_format = "{:,.2f}".format


# Q1.
df=sns.load_dataset("penguins")
print(df.tail().to_string())
print(df.describe())

# Q2.
missing_obs=df.isna().sum().sum()
print(f'The number of missing observations using isna {missing_obs}')
missing_obs=df.isnull().sum().sum()
print(f'The number of missing observations using isnull {missing_obs}')
df=df.dropna()
missing_clean=df.isna().sum().sum()
print(f"After cleaning the number of missing observations are: {df.isna().sum().sum()}")

# Q3.
sns.set_style('darkgrid')
plt.figure()
sns.histplot(data=df,x=df['flipper_length_mm'],kde=True)
plt.title('Q3.')
plt.xlabel('flipper_length_mm')
plt.ylabel('# of observations')
plt.tight_layout()
plt.show()
#
# Q4.
plt.figure()
sns.histplot(data=df,x=df['flipper_length_mm'],kde=True,binwidth=3)
plt.title('Q4.')
plt.xlabel('flipper_length_mm')
plt.ylabel('# of observations')
plt.tight_layout()
plt.show()
#
# Q5.
plt.figure()
sns.histplot(data=df,x=df['flipper_length_mm'],kde=True,bins=30)
plt.title('Q5.')
plt.xlabel('flipper_length_mm')
plt.ylabel('# of observations')
plt.tight_layout()
plt.show()
#
# Q6.
plt.figure()
sns.displot(x=df['flipper_length_mm'],hue=df['species'])
plt.title('Q6.')
plt.xlabel('flipper_length_mm')
plt.ylabel('# of observations')
plt.tight_layout()
plt.show()
#
# Q7.
plt.figure()
sns.displot(x=df['flipper_length_mm'],hue=df['species'],element='step')
plt.title('Q7.')
plt.xlabel('flipper_length_mm')
plt.ylabel('# of observations')
plt.tight_layout()
plt.show()
#
# # Q8.
sns.displot(x=df['flipper_length_mm'],hue=df['species'],element='step',multiple='stack')
plt.title('Q8.')
plt.xlabel('flipper_length_mm')
plt.ylabel('# of observations')
plt.tight_layout()
plt.show()
#
# Q9.
sns.displot(x=df['flipper_length_mm'],hue=df['sex'],multiple='dodge')
plt.title('Q9.')
plt.xlabel('flipper_length_mm')
plt.ylabel('# of samples')
plt.show()
#
# Q10
fig,ax=plt.subplots(1,2)
sns.histplot(x=df[df['sex']=='Male']['flipper_length_mm'],ax=ax[0],label='Male')
sns.histplot(x=df[df['sex']=='Female']['flipper_length_mm'],ax=ax[1],label='Female')
ax[0].set_xlabel('flipper_length_mm')
ax[0].set_ylabel('# of samples')
ax[1].set_xlabel('flipper_length_mm')
ax[1].set_ylabel('# of samples')
fig.suptitle('Q10.')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()
#
# Q11
sns.displot(x=df['flipper_length_mm'],hue=df['species'],stat='density')
plt.title("Q11")
plt.xlabel('Species')
plt.ylabel('Density')
plt.show()
# # #
# # Q12
sns.displot(x=df['flipper_length_mm'],hue=df['sex'],stat='density')
plt.title("Q12")
plt.xlabel('Sex')
plt.ylabel('Density')
plt.show()
# # #
# # Q13
sns.displot(x=df['flipper_length_mm'],hue=df['species'],stat='probability',kde=True)
plt.title("Q13")
plt.xlabel('Species')
plt.ylabel('Density')
plt.show()
# #
# # # Q14
sns.displot(x=df['flipper_length_mm'],hue=df['species'],kind='kde')
plt.title("Q14")
plt.xlabel("flipper_length_mm")
plt.ylabel("Distribution")
plt.show()
# #
# # # Q15
sns.displot(x=df['flipper_length_mm'],hue=df['sex'],kind='kde')
plt.title("Q15")
plt.xlabel("flipper_length_mm")
plt.ylabel("Distribution")
plt.show()
# #
# # # Q16.
sns.displot(x=df['flipper_length_mm'],hue=df['species'],kind='kde',multiple='stack')
plt.title("Q16")
plt.xlabel("flipper_length_mm")
plt.ylabel("Distribution")
plt.show()
#
# # Q17
sns.displot(x=df['flipper_length_mm'],hue=df['sex'],kind='kde',multiple='stack')
plt.title("Q17")
plt.xlabel("flipper_length_mm")
plt.ylabel("Distribution")
plt.show()
# #
# # # Q18
sns.displot(x=df['flipper_length_mm'],hue=df['species'],kind='kde',fill=True)
plt.title("Q18")
plt.xlabel("flipper_length_mm")
plt.ylabel("Distribution")
plt.show()
#
# # Q19
sns.displot(x=df['flipper_length_mm'],hue=df['sex'],kind='kde',fill=True)
plt.title("Q19")
plt.xlabel("flipper_length_mm")
plt.ylabel("Distribution")
plt.show()
# #
# # Q20.
sns.scatterplot(x=df['bill_length_mm'],y=df['bill_depth_mm'],color='blue')
sns.regplot(x=df['bill_length_mm'],y=df['bill_depth_mm'],color='red')
plt.title("Q20")
plt.xlabel("bill_length_mm")
plt.ylabel("bill_depth_mm")
plt.show()
#
# # Q21.
sns.countplot(x=df['island'],hue=df['species'])
plt.title("Q21")
plt.xlabel("Island")
plt.ylabel("# of penguins")
plt.show()
#
# # Q22.
sns.countplot(x=df['island'],hue=df['sex'])
plt.title("Q22")
plt.xlabel("Gender")
plt.ylabel("# of penguins")
plt.show()

# Q23.
sns.displot(x=df['bill_length_mm'],y=df['bill_depth_mm'],hue=df['sex'],kind='kde',fill=True)
plt.title("Q23")
plt.xlabel("bill_length_mm")
plt.ylabel("bill_depth_mm")
plt.show()
# #
# # # Q24.
sns.displot(x=df['bill_length_mm'],y=df['flipper_length_mm'],hue=df['sex'],kind='kde',fill=True)
plt.title("Q24")
plt.xlabel("bill_length_mm")
plt.ylabel("flipper_length_mm")
plt.show()
# #
# # Q25.
sns.displot(x=df['flipper_length_mm'],y=df['bill_depth_mm'],hue=df['sex'],kind='kde',fill=True)
plt.title("Q25")
plt.xlabel("flipper_length_mm")
plt.ylabel("bill_depth_mm")
plt.show()

# Q26.
fig,ax=plt.subplots(1,3,figsize=(8,4))
sns.kdeplot(x=df['bill_length_mm'],y=df['bill_depth_mm'],hue=df['sex'],fill=True,ax=ax[0])
sns.kdeplot(x=df['bill_length_mm'],y=df['flipper_length_mm'],hue=df['sex'],fill=True,ax=ax[1])
sns.kdeplot(x=df['flipper_length_mm'],y=df['bill_depth_mm'],hue=df['sex'],fill=True,ax=ax[2])
plt.suptitle("Q26")
plt.tight_layout()
plt.show()


# Q27.
sns.displot(x=df['bill_length_mm'],y=df['bill_depth_mm'],hue=df['sex'])
plt.title("Q27")
plt.xlabel("bill_length_mm")
plt.ylabel("bill_depth_mm")
plt.show()
#
# # Q28.
sns.displot(x=df['bill_length_mm'],y=df['flipper_length_mm'],hue=df['sex'])
plt.title("Q28")
plt.xlabel("bill_length_mm")
plt.ylabel("flipper_length_mm")
plt.show()
#
# # Q29.
sns.displot(x=df['flipper_length_mm'],y=df['bill_depth_mm'],hue=df['sex'])
plt.title("Q29")
plt.xlabel("flipper_length_mm")
plt.ylabel("bill_depth_mm")
plt.show()