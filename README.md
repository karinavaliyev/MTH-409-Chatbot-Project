# 🧑‍🌾 Stardew Valley Chatbot

Stardew Valley oyuncuları için Türkçe RAG (Retrieval-Augmented Generation) tabanlı bir soru-cevap chatbotu.

## 📋 Özellikler

- 🤖 **Çoklu Model Desteği**: OpenAI GPT-4o ve Google Gemini arasında geçiş yapabilme
- 📊 **CSV Veritabanı**: Karakter bilgileri ve ekin detayları
- 📖 **PDF Rehber**: Stardew Valley rehber kitabından bilgi çekme
- 🔍 **Akıllı Arama**: ChromaDB ile semantik arama
- 🇹🇷 **Türkçe Yanıtlar**: Tüm cevaplar Türkçe olarak verilir

## 🛠️ Kurulum

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. API Anahtarları

Proje dizininde `.env` dosyası oluşturun:

```env
OPENAI_API_KEY=sk-your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
```

### 3. Veri Hazırlama

`data/` klasörüne aşağıdaki dosyaları ekleyin:
- `characters.csv` - Karakter bilgileri
- `crops.csv` - Ekin bilgileri
- `StardewValley-Guide.pdf` - Rehber kitap (opsiyonel)

### 4. Veri İndeksleme

Verileri vektör veritabanına yüklemek için:

```bash
python ingestion.py
```

### 5. Uygulamayı Başlatma

```bash
streamlit run app.py
```

## 📁 Proje Yapısı

```
MTH-409-Chatbot-Project/
├── app.py              # Streamlit arayüzü
├── ingestion.py        # Veri işleme ve indeksleme
├── requirements.txt    # Python bağımlılıkları
├── .env                # API anahtarları (git'e eklenmemeli)
├── data/
│   ├── characters.csv  # Karakter verileri
│   ├── crops.csv       # Ekin verileri
│   └── StardewValley-Guide.pdf
└── .chroma_stardew/    # Vektör veritabanı (otomatik oluşur)
```

## 🎮 Kullanım

1. Sol panelden kullanmak istediğiniz modeli seçin (OpenAI veya Gemini)
2. Sohbet kutusuna Stardew Valley hakkında sorularınızı yazın
3. Chatbot, veritabanından ilgili bilgileri bulup Türkçe yanıt verecektir
4. "📚 Kaynaklar" bölümünden yanıtın hangi kaynaklardan geldiğini görebilirsiniz

## 💡 Örnek Sorular

- "Abigail'e ne hediye etmeliyim?"
- "Patates ne kadar sürede yetişir?"
- "Yaz mevsiminde hangi ekinleri ekebilirim?"
- "Emily'nin doğum günü ne zaman?"

## 📝 Lisans

Bu proje MTH-409 dersi kapsamında eğitim amaçlı geliştirilmiştir.
