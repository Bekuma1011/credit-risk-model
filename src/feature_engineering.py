from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from transformers import TimeFeatureExtractor

def build_feature_pipeline(numerical_cols, categorical_cols, datetime_col):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    full_pipeline = Pipeline([
        ('time_features', TimeFeatureExtractor(datetime_col=datetime_col)),
        ('preprocess', preprocessor)
    ])

    return full_pipeline
