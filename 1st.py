import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

interactions = pd.read_csv('interactions.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Fix dates
for df in [interactions, train, test]:
    df['service_date'] = pd.to_datetime(df['service_date'], format='%d-%m-%Y')

# Prepare metadata maps
origin_cols = ['origin_hub_id', 'origin_region', 'origin_hub_tier']
dest_cols = ['destination_hub_id', 'destination_region', 'destination_hub_tier']

origin_meta = interactions[origin_cols].drop_duplicates()
origin_meta.columns = ['origin_hub_id', 'origin_region', 'origin_tier']

dest_meta = interactions[dest_cols].drop_duplicates()
dest_meta.columns = ['destination_hub_id', 'dest_region', 'dest_tier']

def enrich_data(df):
    df = df.merge(origin_meta, on='origin_hub_id', how='left')
    df = df.merge(dest_meta, on='destination_hub_id', how='left')
    return df.fillna('Unknown')

train = enrich_data(train)
test = enrich_data(test)

def add_time_features(df):
    df = df.copy()
    df['day'] = df.service_date.dt.day
    df['month'] = df.service_date.dt.month
    df['year'] = df.service_date.dt.year
    df['dayofweek'] = df.service_date.dt.dayofweek
    df['is_weekend'] = (df.dayofweek >= 5).astype(int)
    df['quarter'] = df.service_date.dt.quarter
    return df

train = add_time_features(train)
test = add_time_features(test)

# Encoding
cat_cols = ['origin_region', 'origin_tier', 'dest_region', 'dest_tier']
for col in cat_cols:
    le = LabelEncoder()
    full_list = pd.concat([train[col], test[col]]).astype(str)
    le.fit(full_list)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

features = [
    'origin_hub_id', 'destination_hub_id', 'origin_region', 'origin_tier',
    'dest_region', 'dest_tier', 'day', 'month', 'year', 
    'dayofweek', 'is_weekend', 'quarter'
]
target = 'final_service_units'

X = train[features]
y = train[target]
X_test = test[features]

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42
)

model.fit(X, y)

preds = model.predict(X_test)
preds = np.maximum(preds, 0)

sub = pd.DataFrame({'service_key': test['service_key'], 'final_service_units': preds})
sub.to_csv('final_submission.csv', index=False)


check this code by human witten or AI written by percentage







