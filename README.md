# 🧑‍🌾 Stardew Valley Chatbot

Stardew Valley oyuncuları için Türkçe RAG (Retrieval-Augmented Generation) tabanlı bir soru-cevap chatbotu.

## 📋 Özellikler

- 🤖 **Çoklu Model Desteği**: OpenAI GPT-4o ve Google Gemini arasında geçiş yapabilme
- 📊 **CSV Veritabanı**: Karakter bilgileri ve ekin detayları
- 📖 **PDF Rehber**: Stardew Valley rehber kitabından bilgi çekme
- 🔍 **Akıllı Arama**: ChromaDB ile semantik arama
- 🇹🇷 **Türkçe Yanıtlar**: Tüm cevaplar Türkçe olarak verilir
- 📈 **RAGAS Değerlendirmesi**: Faithfulness, Context Recall ve Answer Relevancy metrikleriyle model performans analizi

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
- `StardewValley-Guide.pdf` - Rehber kitap

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
├── app.py                      # Streamlit arayüzü
├── ingestion.py                # Veri işleme ve indeksleme
├── ragas_eval.py               # Tek model RAGAS değerlendirmesi
├── evaluate_compare.py         # Çoklu model karşılaştırmalı değerlendirme
├── requirements.txt            # Python bağımlılıkları
├── .env                        # API anahtarları (git'e eklenmemeli)
├── data/
│   ├── characters.csv          # Karakter verileri
│   ├── crops.csv               # Ekin verileri
│   └── StardewValley-Guide.pdf # Oyun rehberi
├── .chroma_stardew/            # Vektör veritabanı (otomatik oluşur)
├── ragas_report.csv            # Tek model değerlendirme raporu
└── model_comparison_report.csv # Model karşılaştırma raporu
```

## 📊 RAGAS Değerlendirmesi

Proje, RAG sisteminin kalitesini ölçmek için RAGAS framework'ünü kullanır.

### Tek Model Değerlendirmesi

```bash
python ragas_eval.py
```

### Model Karşılaştırması (OpenAI vs Gemini)

```bash
python evaluate_compare.py
```

### Kullanılan Metrikler

| Metrik | Açıklama |
|--------|----------|
| **Faithfulness** | Cevabın, verilen bağlama (context) ne kadar sadık olduğunu ölçer |
| **Context Recall** | Ground truth'un bağlamdan ne kadar çıkarılabildiğini ölçer |
| **Answer Relevancy** | Cevabın soruyla ne kadar alakalı olduğunu ölçer |

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
- "Missing Bundle için hangi malzemeler gerekir?"

## 📝 Lisans

Bu proje MTH-409 dersi kapsamında eğitim amaçlı geliştirilmiştir.

