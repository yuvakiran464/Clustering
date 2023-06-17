#!/usr/bin/env python
# coding: utf-8

# # Assignment 07: Clustering - 02

# #### Dataset: crime_data.csv

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[6]:


# Loading Dataset
data = pd.read_csv('Datasets/crime_data.csv')
data


# ## Data Preprocessing and EDA

# In[12]:


data = data.rename({'Unnamed: 0':'State'}, axis = 1)


# In[13]:


data.info()


# In[14]:


data.isna().sum()


# # Agglomerative Clustering

# In[15]:


from sklearn.preprocessing import MinMaxScaler

# Normalizing Dataset

scaler = MinMaxScaler()

scaler_df = scaler.fit_transform(data.iloc[:,1:])
print(scaler_df)


# In[18]:


# Create Dendrograms
plt.figure(figsize=(10, 7))  
dendograms=sch.dendrogram(sch.linkage(scaler_df,'complete'))


# In[20]:


# Creating clusters
H_clusters=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
H_clusters


# In[21]:


# Using data normalized by MinMaxScaler 
y=pd.DataFrame(H_clusters.fit_predict(scaler_df),columns=['clustersid'])
y['clustersid'].value_counts()


# In[22]:


# Adding clusters to dataset
data['clustersid_HC']=H_clusters.labels_
data


# In[25]:


data.groupby('clustersid_HC').agg(['mean']).reset_index()


# In[27]:


# Plotting barplot using groupby method to get visualization of how states in each cluster
fig, ax = plt.subplots(figsize=(10, 6))
data.groupby(['clustersid_HC']).count()['State'].plot(kind='bar')
plt.ylabel('States')
plt.title('Hierarchical Clustering',fontsize='large',fontweight='bold')
ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')
ax.set_ylabel('States', fontsize='large', fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[28]:


# silhouette_score of AgglomerativeClustering
from sklearn.metrics import silhouette_score


# In[29]:


sil_score= silhouette_score(scaler_df, H_clusters.labels_)
sil_score


# In[33]:


# States in cluster #0 
data[data['clustersid_HC']==0]


# In[31]:


# States in cluster #1 
data[data['clustersid_HC']==1]


# In[32]:


# States in cluster #2 
data[data['clustersid_HC']==2]


# In[34]:


# States in cluster #3 
data[data['clustersid_HC']==3]


# # 

# # K-MEANS Clustering

# In[39]:


# Import Library
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# #### The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:

# In[40]:


# Using data normalized by MinMaxScaler
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaler_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# #### From above Scree plot, optimum number of clusters can be selected equal to 4

# In[41]:


#Build Cluster algorithm

KM_clusters = KMeans(4, random_state=42)
KM_clusters.fit(scaler_df)


# In[42]:


y=pd.DataFrame(KM_clusters.fit_predict(scaler_df),columns=['clusterid_Kmeans'])
y['clusterid_Kmeans'].value_counts()


# In[44]:


#Assign clusters to the data set
data['clusterid_Kmeans'] = KM_clusters.labels_
data


# In[45]:


data.groupby('clusterid_Kmeans').agg(['mean']).reset_index()


# In[46]:


# Plotting barplot using groupby method to get visualization of how states in each cluster
fig, ax = plt.subplots(figsize=(10, 6))
data.groupby(['clusterid_Kmeans']).count()['State'].plot(kind='bar')
plt.ylabel('States')
plt.title('KMeans Clustering',fontsize='large',fontweight='bold')
ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')
ax.set_ylabel('States', fontsize='large', fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[47]:


# States in cluster #0 
data[data['clusterid_Kmeans']==0]


# In[48]:


# States in cluster #1 
data[data['clusterid_Kmeans']==1]


# In[49]:


# States in cluster #2 
data[data['clusterid_Kmeans']==2]


# In[50]:


# States in cluster #1 
data[data['clusterid_Kmeans']==2]


# # 

# # DBSCAN

# In[51]:


from sklearn.cluster import DBSCAN


# #### We will try for different values of eps and mn_samples

# In[77]:


EPS = [0.22, 0.24, 0.26,0.28, 0.30]

for n in EPS:
    dbscan = DBSCAN(eps=n, min_samples=4)
    dbscan.fit(scaler_df)
    y=pd.DataFrame(dbscan.fit_predict(scaler_df),columns=['clusterid_DBSCAN'])
    print(f'For eps = {n}','\n',y['clusterid_DBSCAN'].value_counts())
    # silhouette score
    sil_score= silhouette_score(scaler_df, dbscan.labels_)
    print(f'For eps silhouette score = {n}','\n', sil_score)


# #### When we have value of epsilon = 0.28, we are getting 3 clusters  silhouette score is more as compared to other dbscan models.
# #### -1 shows the noisy data points

# In[78]:


dbscan = DBSCAN(eps=0.28, min_samples=4)
dbscan.fit(scaler_df)


# In[80]:


data['clusterid_DBSCAN'] = dbscan.labels_
data.head()


# In[81]:


data.groupby('clusterid_DBSCAN').agg(['mean']).reset_index()


# In[82]:


# Plotting barplot using groupby method to get visualization of how many row no. in each cluster

fig, ax = plt.subplots(figsize=(10, 6))
data.groupby(['clusterid_DBSCAN']).count()['State'].plot(kind='bar')
plt.ylabel('ID Counts')
plt.title('DBSCAN Clustering',fontsize='large',fontweight='bold')
ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')
ax.set_ylabel('States', fontsize='large', fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[83]:


# States in cluster #0 
data[data['clusterid_DBSCAN']==0]


# In[84]:


# States in cluster #1 
data[data['clusterid_DBSCAN']==1]


# In[85]:


# States in cluster #2 
data[data['clusterid_DBSCAN']==2]


# In[86]:


# States in cluster #1 
data[data['clusterid_DBSCAN']==-1]

