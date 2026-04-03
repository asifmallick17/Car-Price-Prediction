import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

df = pd.read_csv("Dataset/final_data.csv")

print("Columns:\n", df.columns)
print("\nMissing values:\n", df.isnull().sum())

df.rename(columns={'ownsership': 'ownership'}, inplace=True)

df['car_name'] = df['car_name'].apply(lambda x: str(x).split()[0])

num_cols_convert = [
    'registration_year','seats','kms_driven',
    'manufacturing_year','mileage(kmpl)',
    'engine(cc)','max_power(bhp)','torque(Nm)'
]

for col in num_cols_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Feature engineering
df['car_age'] = 2026 - df['manufacturing_year']

# Drop unnecessary column
df.drop(['manufacturing_year'], axis=1, inplace=True)


# REMOVE OUTLIERS
Q1 = df['price(in lakhs)'].quantile(0.25)
Q3 = df['price(in lakhs)'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['price(in lakhs)'] >= Q1 - 1.5 * IQR) & 
        (df['price(in lakhs)'] <= Q3 + 1.5 * IQR)]

X = df.drop("price(in lakhs)", axis=1)
y = df["price(in lakhs)"]

# Log transform target
y = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = [col for col in X.columns if X[col].dtype != 'O']
cat_cols = [col for col in X.columns if X[col].dtype == 'O']


# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # handles NaN
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # handles NaN
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# Reverse log transform
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

r2 = r2_score(y_test_actual, y_pred_actual)
mse = mean_squared_error(y_test_actual, y_pred_actual)

print("R2 Score:", r2)
print("MSE:", mse)

joblib.dump(pipeline, "car_price_model.pkl")

print("Model saved successfully!")