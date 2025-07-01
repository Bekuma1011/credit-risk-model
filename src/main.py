import pandas as pd
from feature_engineering import build_feature_pipeline
from transformers import AggregateCustomerFeatures

# Load data
df = pd.read_csv('../data/raw/data(1).csv')
print("Columns in DataFrame:", df.columns.tolist())

# 1. Create aggregate features separately
agg = AggregateCustomerFeatures().fit_transform(df)

# 2. Merge aggregate features back into original df
df = df.merge(agg, on='CustomerId', how='left')
pd.DataFrame(df).to_csv('../data/processed/test.csv', index=False)
# 3. Define features to include in the pipeline
numerical_cols = ['Amount', 'Value', 'TotalAmount', 'AvgAmount', 'Count', 'StdAmount']
categorical_cols = ['ProductCategory', 'ChannelId', 'CountryCode', 'CurrencyCode']
datetime_col = 'TransactionStartTime'

# 4. Build and apply pipeline
pipeline = build_feature_pipeline(numerical_cols, categorical_cols, datetime_col)
processed_data = pipeline.fit_transform(df)

# 5. Save output
pd.DataFrame(processed_data).to_csv('../data/processed/processed_transactions.csv', index=False)
