import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load data
df = pd.read_csv('../data/raw/data(1).csv')

# 2. Convert TransactionStartTime to datetime (from UTC-Z format)
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True).dt.tz_localize(None)

# 3. Define snapshot date (the reference point for recency)
snapshot_date = datetime(2025, 7, 1)

# 4. Calculate RFM metrics
rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
}).rename(columns={
    'TransactionStartTime': 'Recency',
    'TransactionId': 'Frequency',
    'Amount': 'Monetary'
}).reset_index()

# 5. Scale RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# 6. Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 7. Identify the high-risk cluster (high recency, low freq & monetary)
centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Recency', 'Frequency', 'Monetary'])
high_risk_cluster = centers.sort_values(
    by=['Recency', 'Frequency', 'Monetary'],
    ascending=[False, True, True]
).index[0]

# 8. Assign proxy target
rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

# 9. Merge target back into main DataFrame
df = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# 10. Save the output
df.to_csv('../data/processed/data_with_target.csv', index=False)

print(" Done. File saved as: ../data/processed/data_with_target.csv")
