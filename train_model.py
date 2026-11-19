import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load data
df = pd.read_csv("movies.csv", encoding="utf-8-sig")
print("Shape:", df.shape)
print("Missing values before:")
print(df.isna().sum())

# 2. Features and target
num_cols = ["Duration_Min", "Budget_Crore_INR", "Popularity_Index"]
cat_cols = ["Genre"]
X = df[num_cols + cat_cols].copy()
X["Genre"] = X["Genre"].astype(object)
y = df["Rating_10"]

# 3. Handle missing values: median for numbers, most frequent for Genre
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

model = Pipeline([
    ("prep", preprocess),
    ("reg", RandomForestRegressor(n_estimators=100, random_state=42)),
])

# 4. Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print("R2 :", round(r2_score(y_test, pred), 3))

# 5. Save model
joblib.dump(model, "movie_model.pkl")
print("Model saved as movie_model.pkl")