import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score


data = load_diabetes()
X = data.data[:, np.newaxis, 2]
y = data.target


model = joblib.load("/data/model.pkl")


y_pred = model.predict(X)


accuracy = r2_score(y, y_pred)
print(f"Model accuracy (R^2 score): {accuracy:.2f}")


plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_pred, color="red", label="Predicted")
plt.title("Linear Regression Predictions")
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.legend()
plt.savefig("/data/predictions.png")
print("Plot saved to /data/predictions.png")
