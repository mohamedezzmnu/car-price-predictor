import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv(sys.argv[1])
outdir = sys.argv[2]

df = df.drop(columns=["ID", "Model", "Color"])
df = df.drop_duplicates()

# عمود Levy فيه "-" معناها قيمة ناقصة
df["Levy"] = pd.to_numeric(df["Levy"].replace("-", np.nan))
df["Levy"] = df["Levy"].fillna(df["Levy"].median())

df["Mileage"] = df["Mileage"].str.replace(" km", "", regex=False).astype(float)
df["Turbo"] = df["Engine volume"].str.contains("Turbo").astype(int)
df["Engine volume"] = df["Engine volume"].str.replace(" Turbo", "", regex=False).astype(float)

# اتقرأت غلط كتاريخ (04-May = 4 أبواب، 02-Mar = 2 باب)
df["Doors"] = df["Doors"].map({"04-May": 4, "02-Mar": 2, ">5": 5})

df["Leather interior"] = (df["Leather interior"] == "Yes").astype(int)
df["Wheel"] = (df["Wheel"] == "Right-hand drive").astype(int)
df["Car age"] = 2020 - df["Prod. year"]
df = df.drop(columns="Prod. year")

# أسعار تافهة (أقل من 2000) أو ضخمة جداً، وكيلومترات صفر أو مبالغ فيها
df = df[(df["Price"] >= 2000) & (df["Price"] <= 100000)]
df = df[(df["Mileage"] > 0) & (df["Mileage"] <= 500000)]
df = df[df["Car age"] <= 40]

# الكيلومترات منحرفة فبنعمل log
df["Mileage"] = np.log1p(df["Mileage"])
print("الصفوف بعد التنضيف:", len(df))

df = pd.get_dummies(df, columns=["Manufacturer", "Category", "Fuel type", "Gear box type", "Drive wheels"], drop_first=True, dtype=int)

# السعر منحرف فبنعمله log
y = np.log1p(df["Price"])
X = df.drop(columns="Price")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ["Levy", "Engine volume", "Mileage", "Cylinders", "Airbags", "Car age", "Doors"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

model = LinearRegression()
model.fit(X_train, y_train)

for name, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
    pred = model.predict(Xs)
    real, real_pred = np.expm1(ys), np.expm1(pred)
    print(name)
    print("  R2 (log):", round(r2_score(ys, pred), 3))
    print("  R2 (price):", round(r2_score(real, real_pred), 3))
    print("  MAE:", round(mean_absolute_error(real, real_pred)))
    print("  RMSE:", round(np.sqrt(mean_squared_error(real, real_pred))))

coef = pd.Series(model.coef_, index=X.columns)
print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(10).round(3))

pred = model.predict(X_test)
res = y_test - pred
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(y_test, pred, s=4, alpha=0.4)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r")
ax[0].set_title("Actual vs Predicted")
ax[1].scatter(pred, res, s=4, alpha=0.4)
ax[1].axhline(0, color="r")
ax[1].set_title("Residuals vs Predicted")
ax[2].hist(res, bins=50)
ax[2].set_title("Residuals")
plt.tight_layout()
plt.savefig(outdir + "/model_diagnostics.png", dpi=110)
