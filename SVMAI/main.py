import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import joblib  # Modeli kaydetmek ve yüklemek için joblib ekliyoruz

# NLTK stopwords yükleniyor
nltk.download('stopwords')

# Veri setini yükleyin
veri_seti_yolu = './test.csv'
veri = pd.read_csv(veri_seti_yolu)

# 'label' sütununda 5 değeri olan satırları temizleyin
veri = veri[veri['label'] != 5]

# Veriyi 'text' ve 'label' sütunlarına ayırın
X = veri['text']
y = veri['label']

# Stopwords'i kaldırmak için bir fonksiyon tanımlayın
stop_words = set(stopwords.words('english'))

def temizle(text):
    text = text.lower()  # Küçük harfe dönüştür
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Sadece harf ve boşluk bırak
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Stopwords'leri kaldır
    return text

# 'text' verilerini temizleyin
X = X.apply(temizle)

# Veriyi eğitim ve test olarak ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# TF-IDF vektörizasyonu ve SVM modelini bir pipeline içine alalım
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),  # 2-gram ekledik, özellikleri artırır
    ('svm', SVC(kernel='linear', random_state=42))
])

# GridSearchCV ile parametreleri optimize edelim
param_grid = {
    'svm__C': [0.1, 1, 10, 100],  # C parametresi için değerler
    'svm__gamma': ['scale', 'auto'],  # Gamma parametresi
    'svm__kernel': ['linear', 'rbf', 'poly'],  # Kernel seçenekleri
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)

# Modeli eğitiyoruz
grid_search.fit(X_train, y_train)

# En iyi parametrelerle eğitilen modeli alıyoruz
best_model = grid_search.best_estimator_

# Modeli kaydediyoruz
joblib.dump(best_model, 'duygu_modeli.joblib')  # Model dosyaya kaydediliyor

# Test seti üzerinde tahmin yapın
y_pred = best_model.predict(X_test)

# Sonuçları yazdırın
dogruluk = accuracy_score(y_test, y_pred) * 100  # Yüzdesel doğruluk
print(f"Doğruluk: %{dogruluk:.2f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Kullanıcıdan tweet alıp tahmin yapma
while True:
    yeni_tweet = input("Bir tweet girin (çıkmak için 'q' yazın): ")
    if yeni_tweet.lower() == 'q':
        break
    temizlenmis_tweet = temizle(yeni_tweet)
    
    # Eğitilen modeli yükleyip kullanıyoruz
    loaded_model = joblib.load('duygu_modeli.joblib')  # Kaydedilen modeli yüklüyoruz
    tahmin = loaded_model.predict([temizlenmis_tweet])
    
    # Duygu sınıfına göre çıktı verir
    duygular = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear']
    print(f"Bu tweet'in duygu durumu: {duygular[tahmin[0]]}")
