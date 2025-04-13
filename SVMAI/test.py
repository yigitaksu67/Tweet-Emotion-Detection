import joblib  # Modeli yüklemek için joblib
import nltk
from nltk.corpus import stopwords

# NLTK stopwords yükleniyor
nltk.download('stopwords')

# Stopwords'i kaldırmak için bir fonksiyon tanımlayın
stop_words = set(stopwords.words('english'))

def temizle(text):
    text = text.lower()  # Küçük harfe dönüştür
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Sadece harf ve boşluk bırak
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Stopwords'leri kaldır
    return text

# Daha önce kaydedilen model dosyasını yükleyin
model = joblib.load('duygu_modeli.joblib')  # Burada 'duygu_modeli.joblib' modelin kaydedildiği dosya ismi

# Kullanıcıdan tweet alıp tahmin yapma
while True:
    yeni_tweet = input("Bir tweet girin (çıkmak için 'q' yazın): ")
    if yeni_tweet.lower() == 'q':  # 'q' yazarsa çıkılır
        break
    
    # Tweeti temizleme işlemi
    temizlenmis_tweet = temizle(yeni_tweet)  # 'temizle' fonksiyonunu kullanarak tweeti temizliyoruz
    
    # Modeli kullanarak tahmin yapıyoruz
    tahmin = model.predict([temizlenmis_tweet])  # Modeli kullanarak tahmin alıyoruz
    
    # Duygu sınıfına göre çıktı verir
    duygular = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear']  # Duygu etiketleri
    print(f"Bu tweet'in duygu durumu: {duygular[tahmin[0]]}")
