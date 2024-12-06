import numpy as np
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = load_diabetes()
X = data.data[:, np.newaxis, 2]
y = data.target


X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


joblib.dump(model, "/data/model.pkl")
print("Model trained and saved to /data/model.pkl")
