import pandas as pd
from sklearn.cluster import KMeans
# Step 1: Read Data from CSV
data = pd.read_csv('insurance_leads.csv')
# Step 2: Data Preprocessing (One-Hot Encoding for Categorical Features)
data = pd.get_dummies(data, columns=["Income_Group", "Gender", "Marital_Status"], drop_first=True)
# Step 3: Choose the Number of Clusters
k = 4  # You can adjust this value based on your analysis or objectives
# Step 4: Apply K-Means Clustering
kmeans = KMeans(n_clusters=k, random_state=0)
data['Cluster'] = kmeans.fit_predict(data)
# Step 5: Map numeric cluster labels to meaningful labels
cluster_labels = {
    0: "Cluster A",
    1: "Cluster B",
    2: "Cluster C",
    3: "Cluster D"
}
data['Cluster_Label'] = data['Cluster'].map(cluster_labels)
# Step 6: Print the resulting DataFrame with cluster labels
print(data)