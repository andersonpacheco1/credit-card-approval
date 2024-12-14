import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

import joblib

from pathlib import Path

from features import DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMaxWithFeatNames, OverSample

data_path = Path('data/processed/clean_credit_record.csv')

credit = pd.read_csv(data_path)

seed = 1561651

def pipeline(df):

  pipeline = Pipeline([
      ('feature_dropper', DropFeatures()),
      ('min_max_scaler', MinMaxWithFeatNames()),
      ('one_hot_encoder', OneHotEncodingNames()),
      ('ordinal_encoder', OrdinalFeature()),
      ('over_sampler', OverSample())
  ])

  df_pipeline = pipeline.fit_transform(df)

  return df_pipeline

train, test = train_test_split(credit, train_size=0.8, random_state=seed)

model_train = pipeline(train)
X_train, y_train = model_train.loc[:, model_train.columns != 'Mau'], model_train['Mau']

model_test = pipeline(test)
X_test, y_test = model_test.loc[:, model_test.columns != 'Mau'], model_test['Mau']

xgb = GradientBoostingClassifier()
xgb.fit(X_train, y_train)

output_dir = Path('models')
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / 'xgb.pkl'
joblib.dump(xgb, model_path)