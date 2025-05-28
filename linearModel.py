import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Veri oluşturuyorum 
np.random.seed(42)
tarihler = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
n = len(tarihler)

#features
yil = tarihler.year
ay = tarihler.month
kampanya = np.random.choice([0, 1], size=n, p=[0.9, 0.1])  # Kampanya günleri
doviz = 15 + np.random.normal(0, 1, size=n) + (tarihler.year - 2020) * 0.5

# Mevsimsellik etkisi (sin, cos ile)
sin_ay = np.sin(2 * np.pi * ay / 12)
cos_ay = np.cos(2 * np.pi * ay / 12)

# Satış verisi (hedef değişken)
satis = (
    100 +
    yil * 5 +
    kampanya * 100 +
    doviz * 10 +
    sin_ay * 20 +
    np.random.normal(0, 20, size=n)
)

# DataFrame 
df = pd.DataFrame({
    "Tarih": tarihler,
    "Yil": yil,
    "Kampanya": kampanya,
    "Doviz": doviz,
    "sin_ay": sin_ay,
    "cos_ay": cos_ay,
    "Satis": satis
})

# Eğitim ve test verisi
X = df[["Yil", "Kampanya", "Doviz", "sin_ay", "cos_ay"]]
y = df["Satis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model oluşturutorum  burda eğitiyorum
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin burada yapılıypr
y_pred = model.predict(X_test)

# Değerlendirme yapılıyor
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
print("Katsayılar:", dict(zip(X.columns, model.coef_)))

# Grafik
plt.figure(figsize=(12, 5))
plt.plot(df["Tarih"].iloc[-len(y_test):], y_test.values, label="Gerçek Satış")
plt.plot(df["Tarih"].iloc[-len(y_test):], y_pred, label="Tahmin", linestyle="--")
plt.xlabel("Tarih")
plt.ylabel("Satış")
plt.title("📈 Linear Regression ile Satış Tahmini")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
