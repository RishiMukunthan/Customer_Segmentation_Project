from fancyimpute import KNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

# load dataset
data = pd.read_csv('credit-card-data.csv')
data.drop('CUST_ID',axis=1,inplace=True)

#Apply KNN imputation algorithm
data = pd.DataFrame(KNN(k = 3).fit_transform(data), columns = data.columns)

#Variables with Missing Value percentage
data.apply(lambda x: sum(x.isnull()/len(data))*100)

data.to_csv('credit_card_knn_imputed.csv', index = False)

data['MA_PURCHASE'] = data['PURCHASES']/data['TENURE']
data['MA_CASH_ADVANCE'] = data['CASH_ADVANCE']/data['TENURE']
data['LIMIT_USAGE'] = data['BALANCE']/data['CREDIT_LIMIT']
data['PAY_MINPAY_RATIO'] = data['PAYMENTS']/data['MINIMUM_PAYMENTS']

#drop purchases,cash_advance,tenure(less variability),Balance,CreditLimit

def purchase_type(data):
    if (data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']==0):
        return 'NONE'
    if (data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']>0):
         return 'BOTH'
    if (data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']==0):
        return 'ONE_OFF'
    if (data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']>0):
        return 'INSTALLMENT'

data['PURCHASE_TYPE']=data.apply(purchase_type,axis=1)

data = pd.concat([data,pd.get_dummies(data['PURCHASE_TYPE'],drop_first=True)],axis=1)
data_profile = data.copy()
data.drop(columns=['PURCHASE_TYPE','PURCHASES','CASH_ADVANCE','TENURE','BALANCE','CREDIT_LIMIT'
                  ,'PAYMENTS','MINIMUM_PAYMENTS'],axis=1,inplace=True)


# log tranformation
data_logT =data[['BALANCE_FREQUENCY', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'PURCHASES_FREQUENCY',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'PRC_FULL_PAYMENT','MA_PURCHASE', 'MA_CASH_ADVANCE', 'LIMIT_USAGE',
       'PAY_MINPAY_RATIO']].applymap(lambda x: np.log(x+1))

data_logT = pd.concat([data_logT,data[['NONE','ONE_OFF','INSTALLMENT']]],axis=1)

from sklearn.preprocessing import  MinMaxScaler
mm=MinMaxScaler()
data_scaled=mm.fit_transform(data_logT)

data_profile.groupby('PURCHASE_TYPE').apply(lambda x: np.mean(x['MA_PURCHASE'])).plot.barh(color=['m','g','b','y'])
plt.title('Average Monthly Purchase by Purchase Types')
plt.savefig("Monthly_Average_Purchase_by_PurchaseType.png")
plt.show()

data_profile.groupby('PURCHASE_TYPE').apply(lambda x: np.mean(x['MA_CASH_ADVANCE'])).plot.barh(color=['m','g','b','y'])
plt.title('Average Cash Advance amount by Purchase Types')
plt.savefig("Cash_Advance_by_PurchaseType.png")
plt.show()

data_profile.groupby('PURCHASE_TYPE').apply(lambda x: np.mean(x['PAY_MINPAY_RATIO'])).plot.barh(color=['m','g','b','y'])
plt.title('Mean Payment_MinPayment Ratio by Purchase Type')
plt.savefig("Pay_MinPay_Ratio_by_PurchaseType.png")
plt.show()

from sklearn.decomposition import PCA

var_ratio={}
for n in range(2,17):
    pc=PCA(n_components=n)
    data_pca=pc.fit(data_scaled)
    var_ratio[n]=sum(data_pca.explained_variance_ratio_)
    
var_ratio

pd.Series(var_ratio).plot(title='Number of Principal Components vs Explained Variance')
plt.savefig("PCA Explained Variance Graph.png")
plt.show()

pca_final=PCA(n_components=4).fit(data_scaled)
PC_data=pca_final.fit_transform(data_scaled)
PC_data=pd.DataFrame(PC_data)

# Factor Analysis : variance explained by each component- 
pd.Series(pca_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(1,5)])

PCA_insight = pd.DataFrame(pca_final.components_.T, columns=['PC_' +str(i) for i in range(1,5)],index=data.columns)
PCA_insight.abs().sort_values('PC_1',ascending=False)

PCA_insight.abs().sort_values('PC_2',ascending=False)

#Load required libraries
from sklearn.cluster import KMeans

#Estimate optimum number of clusters
cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans(num_clusters).fit(PC_data.iloc[:,:])
    cluster_errors.append(clusters.inertia_)
    
#Create dataframe with cluster errors
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

#Plot line chart to visualise number of clusters
#%matplotlib inline  
plt.figure(figsize=(12,6))
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
plt.savefig("No of Clusters vs WCSS.png")
plt.show()

#Implement kmeans
kmeans_model = KMeans(n_clusters = 4,random_state=0).fit(PC_data.iloc[:,:])

idx = np.argsort(kmeans_model.cluster_centers_.sum(axis=1))
lut = np.zeros_like(idx)
lut[idx] = np.arange(4)

kmeans_model.labels_

# Conactenating labels found through Kmeans with data 
data_cluster=pd.concat([data[['MA_PURCHASE','MA_CASH_ADVANCE','LIMIT_USAGE','PAY_MINPAY_RATIO','PURCHASES_TRX'
                             ,'PRC_FULL_PAYMENT','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']],
                        pd.Series(lut[kmeans_model.labels_],name='Group')],axis=1)

data_cluster['Group'].unique()

# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
data_kpi_mean=data_cluster.groupby('Group')\
.apply(lambda x: x[['MA_PURCHASE','MA_CASH_ADVANCE','LIMIT_USAGE','PAY_MINPAY_RATIO',
                   'PURCHASES_TRX','PRC_FULL_PAYMENT','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']].mean()).T
data_kpi_mean

color_map={0:'r',1:'b',2:'g',3:'y'}
label_color=[color_map[l] for l in kmeans_model.labels_]
plt.figure(figsize=(7,7))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(PC_data.iloc[:,0],PC_data.iloc[:,1],c=label_color,cmap='Spectral',alpha=0.1)
plt.savefig("Clustering Results PC1 vs PC2.png")
plt.show()

color_map={0:'r',1:'b',2:'g',3:'y'}
label_color=[color_map[l] for l in kmeans_model.labels_]
plt.figure(figsize=(7,7))
plt.xlabel('One Off Purchases')
plt.ylabel('Installment Purchases')
plt.scatter(data_scaled[:,1],data_scaled[:,2],c=label_color,cmap='Spectral',alpha=0.1)
plt.savefig("Clustering Results OneOff_Purchases vs Installment Purchases.png")
plt.show()

Counts =data_cluster.groupby('Group').apply(lambda x: x['Group'].value_counts())
print (Counts)

fig,ax=plt.subplots(figsize=(5,3))
index=np.arange(len(data_kpi_mean.columns))

MA_CASH_ADVANCE=(data_kpi_mean.loc['MA_CASH_ADVANCE',:].values)

bar_width=.10
b1=plt.bar(index,MA_CASH_ADVANCE,color='r',label='MA_CASH_ADVANCE',width=bar_width)

plt.xlabel("Clusters")
plt.title("Mean MA_Cash_Advance among clusters")
plt.xticks(index, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()
plt.savefig("Segmentation Analysis1.png")
plt.show()

fig,ax=plt.subplots(figsize=(5,3))
index=np.arange(len(data_kpi_mean.columns))

LIMIT_USAGE=(data_kpi_mean.loc['LIMIT_USAGE',:].values)

bar_width=.10
b1=plt.bar(index,LIMIT_USAGE,color='r',label='LIMIT_USAGE',width=bar_width)

plt.xlabel("Cluster")
plt.title("Mean Limit Usage among clusters")
plt.xticks(index, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()
plt.savefig("Segmentation Analysis2.png")
plt.show()

fig,ax=plt.subplots(figsize=(5,3))
index=np.arange(len(data_kpi_mean.columns))

MA_PURCHASE=(data_kpi_mean.loc['MA_PURCHASE',:].values)

bar_width=.10
b1=plt.bar(index,MA_PURCHASE,color='b',label='MA_PURCHASE',width=bar_width)

plt.xlabel("Cluster")
plt.title("Mean MA_Purchase among clusters")
plt.xticks(index, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()
plt.savefig("Segmentation Analysis3.png")
plt.show()

fig,ax=plt.subplots(figsize=(5,3))
index=np.arange(len(data_kpi_mean.columns))

PURCHASES_TRX=(data_kpi_mean.loc['PURCHASES_TRX',:].values)

bar_width=.10
b1=plt.bar(index,PURCHASES_TRX,color='b',label='PURCHASES_TRX',width=bar_width)

plt.xlabel("Clusters")
plt.title("Mean Purchase/Txn among clusters")
plt.xticks(index, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()
plt.savefig("Segmentation Analysis4.png")
plt.show()

fig,ax=plt.subplots(figsize=(5,3))
index=np.arange(len(data_kpi_mean.columns))

PRC_FULL_PAYMENT=(data_kpi_mean.loc['PRC_FULL_PAYMENT',:].values)

bar_width=.10
b1=plt.bar(index,PRC_FULL_PAYMENT,color='b',label='PRC_FULL_PAYMENT',width=bar_width)

plt.xlabel("Clusters")
plt.title("Full_Payment_Due_Percentage among clusters")
plt.xticks(index, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()
plt.savefig("Segmentation Analysis5.png")
plt.show()

fig,ax=plt.subplots(figsize=(5,3))
index=np.arange(len(data_kpi_mean.columns))

PAY_MINPAY_RATIO=(data_kpi_mean.loc['PAY_MINPAY_RATIO',:].values)

bar_width=.10
b1=plt.bar(index,PAY_MINPAY_RATIO,color='b',label='PAY_MINPAY_RATIO',width=bar_width)

plt.xlabel("Clusters")
plt.title("Average Pay_MinPay_Ratio Among Clusters")
plt.xticks(index, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()
plt.savefig("Segmentation Analysis6.png")
plt.show()

fig,ax=plt.subplots(figsize=(8,5))
index=np.arange(len(data_kpi_mean.columns))

ONEOFF_PURCHASES=(data_kpi_mean.loc['ONEOFF_PURCHASES',:].values)
INSTALLMENTS_PURCHASES=(data_kpi_mean.loc['INSTALLMENTS_PURCHASES',:].values)

bar_width=.10
b1=plt.bar(index,ONEOFF_PURCHASES,color='b',label='ONEOFF_PURCHASES',width=bar_width)
b2=plt.bar(index+bar_width,INSTALLMENTS_PURCHASES,color='r',label='INSTALLMENTS_PURCHASES',width=bar_width)

plt.xlabel("Clusters")
plt.title("Purchases - OneOff vs Installment among clusters")
plt.xticks(index, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()
plt.savefig("Segmentation Analysis7.png")
plt.show()

data_label = pd.read_csv('credit_card_knn_imputed.csv')
data_raw = pd.read_csv('credit-card-data.csv')
# Conactenating labels found through Kmeans with data 
data_label=pd.concat([data_raw['CUST_ID'],data_label,
                        pd.Series(lut[kmeans_model.labels_],name='Group')],axis=1)

map_cluster = {0: "Withdrawers", 1: "Installment_Purchasers", 2: "One_Off_Purchasers", 3: "Big_Spenders"}
data_label = data_label.replace({"Group": map_cluster})

data_label.to_csv('Clustering_Output_Python.csv', index = False)