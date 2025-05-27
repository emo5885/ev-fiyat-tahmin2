import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 81 il listesi
iller = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Amasya", "Ankara", "Antalya", "Artvin", "Aydın", "Balıkesir",
    "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli",
    "Diyarbakır", "Edirne", "Elazığ", "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane",
    "Hakkâri", "Hatay", "Isparta", "Mersin", "İstanbul", "İzmir", "Kars", "Kastamonu", "Kayseri", "Kırklareli",
    "Kırşehir", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa", "Kahramanmaraş", "Mardin", "Muğla", "Muş",
    "Nevşehir", "Niğde", "Ordu", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Tekirdağ", "Tokat",
    "Trabzon", "Tunceli", "Şanlıurfa", "Uşak", "Van", "Yozgat", "Zonguldak", "Aksaray", "Bayburt", "Karaman",
    "Kırıkkale", "Batman", "Şırnak", "Bartın", "Ardahan", "Iğdır", "Yalova", "Karabük", "Kilis", "Osmaniye",
    "Düzce"
]

# Şehirlere göre fiyat etkisi
il_fiyat_efekti = {}
base_price = 50000
for i, il in enumerate(iller):
    if il in ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Kocaeli"]:
        il_fiyat_efekti[il] = base_price + 200000 + i * 1000
    elif il in ["Gaziantep", "Konya", "Kayseri", "Sakarya", "Tekirdağ"]:
        il_fiyat_efekti[il] = base_price + 100000 + i * 800
    else:
        il_fiyat_efekti[il] = base_price + i * 500

# Veri üretimi
np.random.seed(42)
random.seed(42)
n = 3000
data = []

for _ in range(n):
    metrekare = np.random.randint(40, 250)
    oda = random.choice([1, 2, 3, 4, 5])
    bina_yasi = np.random.randint(0, 50)
    sehir = random.choice(iller)
    kat = np.random.randint(1, 15)
    site_ici = random.choice([0, 1])
    esyali = random.choice([0, 1])
    asansor = random.choice([0, 1])

    fiyat = (metrekare * 4000) + (oda * 100000) - (bina_yasi * 2500) + il_fiyat_efekti[sehir]
    fiyat += site_ici * 120000 + esyali * 30000 + asansor * 20000
    fiyat += np.random.normal(0, 40000)
    data.append([metrekare, oda, bina_yasi, sehir, kat, site_ici, esyali, asansor, fiyat])

df = pd.DataFrame(data, columns=[
    "Metrekare", "Oda", "Bina_Yasi", "Sehir", "Kat", "Site_Ici", "Esyali", "Asansor", "Fiyat"
])

# Model
X = df.drop("Fiyat", axis=1)
y = df["Fiyat"]

categorical = ["Sehir"]
numeric = ["Metrekare", "Oda", "Bina_Yasi", "Kat", "Site_Ici", "Esyali", "Asansor"]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=150, random_state=42))
])

model.fit(X, y)

# Web Arayüz
st.title("🏠 Ev Fiyat Tahmini Uygulaması")
st.markdown("Türkiye genelinde örnek verilere göre ev fiyat tahmini yapabilirsiniz.")

metrekare = st.slider("Metrekare", 40, 300, 100)
oda = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5])
bina_yasi = st.slider("Bina Yaşı", 0, 50, 10)
sehir = st.selectbox("Şehir", iller)
kat = st.slider("Kaçıncı Kat", 1, 20, 3)
site_ici = st.selectbox("Site İçinde mi?", ["Hayır", "Evet"]) == "Evet"
esyali = st.selectbox("Eşyalı mı?", ["Hayır", "Evet"]) == "Evet"
asansor = st.selectbox("Asansör var mı?", ["Hayır", "Evet"]) == "Evet"

if st.button("Fiyatı Tahmin Et"):
    girdi = pd.DataFrame([{
        "Metrekare": metrekare,
        "Oda": oda,
        "Bina_Yasi": bina_yasi,
        "Sehir": sehir,
        "Kat": kat,
        "Site_Ici": int(site_ici),
        "Esyali": int(esyali),
        "Asansor": int(asansor)
    }])

    tahmin = model.predict(girdi)[0]
    st.success(f"Tahmini Fiyat: {int(tahmin):,} TL")
