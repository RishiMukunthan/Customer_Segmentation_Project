#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

# load dataset
data = pd.read_csv('credit-card-data.csv')

# rows and columns of the data
print(data.shape)

# visualise the dataset
data.head()

features_initial = [data.columns]
features_initial
##We have 18 features, now using data dictionary , understand the features

data.describe().transpose()

#Drop Customer Id Column
data.drop('CUST_ID',axis=1,inplace = True)

#Variables with Missing Value percentage
data.apply(lambda x: sum(x.isnull()/len(data))*100).sort_values(axis=0,ascending=False)

# Let's go ahead and analyse the distributions of the variables

def show_distribution(df, var):
    df = df.copy()
    df[var].hist(bins=30)
    plt.ylabel('Frequency Distribution')
    plt.xlabel(var)
    plt.title(var)
    plt.savefig(f"{var}_Distribution.png")
    plt.show()

for var in data.columns:
    show_distribution(data, var)

def find_outliers(df, var):
    df = df.copy()
    df.boxplot(column=var)
    plt.title(var)
    plt.ylabel(var)
    plt.savefig(f"{var}_Box_Plot.png")
    plt.show()


for var in data.columns:
    find_outliers(data, var)

data_corr = data.corr()
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(20, 20))
matrix = np.triu(data_corr)
sns.heatmap(data_corr[(data_corr >= 0.5) | (data_corr <= -0.4)],annot=True,vmin=-1, vmax=1, 
            center= 0,cmap= 'coolwarm',square=True,mask=matrix)
plt.savefig("Correlation_HeatMap.png")