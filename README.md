# Döllenme Durumu Tarayıcı

Kuluçka sürecinde **candling (ışıkla muayene)** yöntemiyle çekilen yumurta görüntülerini analiz ederek yumurtaları erken aşamada **döllenmiş**, **döllenmemiş** ve **ölü** olarak sınıflandıran, derin öğrenme tabanlı bir görüntü işleme sistemi.

Bu proje, Keskinoğlu'nda yaptığım yaz stajı kapsamında geliştirildi ve şirket içi bir sunucuya kurularak kullanıma hazır hale getirildi.

## Özellikler

- 3 sınıflı yumurta sınıflandırması: `fertile` (döllenmiş), `infertile` (döllenmemiş), `dead` (ölü)
- EfficientNetB0 tabanlı transfer öğrenme modeli
- Test setinde %99 doğruluk (4.275 görüntülük veri seti)
- Grad-CAM ile model kararlarının görselleştirilmesi
- Occlusion testleriyle açıklanabilirlik doğrulaması
- FastAPI tabanlı REST backend
- Türkçe arayüzlü web uygulaması (koyu tema, kenar çubuğu düzeni)
- Çoklu görsel yükleme ve paralel çıkarım
- GPU gerektirmeden CPU üzerinde çalışabilir

## Mimari

```
Kullanıcı (Web Arayüzü)
        |
        v
FastAPI Backend  -->  Ön işleme  -->  EfficientNetB0  -->  Sınıf + güven skoru
        |
        v
Türkçe analiz mesajı + (opsiyonel) Grad-CAM ısı haritası
```

## Kullanılan Teknolojiler

| Katman | Teknoloji |
|---|---|
| Model | PyTorch, EfficientNetB0 |
| Açıklanabilirlik | Grad-CAM, occlusion analizi |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |

## Kurulum

```bash
# 1. Depoyu klonla
git clone https://github.com/Emirtoker10/Dollenmis-Yumurtalar-Tespit-Sistemi.git
cd Dollenmis-Yumurtalar-Tespit-Sistemi

# 2. Sanal ortam oluştur
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. CPU sürümlü PyTorch'u yükle
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Diğer bağımlılıkları yükle
pip install -r requirements.txt
```

## Kullanım

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Ardından tarayıcıdan `http://localhost:8000` adresini aç, bir veya birden fazla yumurta görseli yükle ve analiz sonuçlarını incele.

## Sınıflar

| Sınıf | Anlamı |
|---|---|
| `fertile` | Döllenmiş yumurta |
| `infertile` | Döllenmemiş yumurta |
| `dead` | Gelişimi durmuş / ölü embriyo |

> **Not:** Sınıf sırası eğitimdeki etiket haritasıyla (`LABEL_MAP`) birebir aynı olmalıdır: `["fertile", "infertile", "dead"]`. Sıranın yanlış tanımlanması çıkarım sırasında sonuçların karışmasına neden olur.

## Açıklanabilirlik

Modelin kararlarını yorumlamak için eğitim sonrası **Grad-CAM** entegre edildi (hedef katman: `model.features[-3]`). Isı haritalarında köşelere odaklanan bölgeler görüldüğünde bunun gerçek model davranışı mı yoksa görselleştirme artefaktı mı olduğunu anlamak için **occlusion testleri** uygulandı ve bu odakların görselleştirmeden kaynaklandığı doğrulandı.

## Dağıtım

Sistem, CPU tabanlı PyTorch ile bir Windows sunucusuna kuruldu ve şirket içi kullanıma hazır hale getirildi.

## Yol Haritası

- [ ] Konveyör bandındaki çoklu yumurta tepsileri için nesne tespiti
- [ ] İki aşamalı boru hattı: YOLO ile yumurta konumlandırma + EfficientNetB0 ile sınıflandırma
- [ ] Farklı kuluçkahanelerden gelen verilerle modelin genelleme yeteneğinin artırılması

## Veri ve Model Hakkında Not

Eğitimde kullanılan görüntüler ve eğitilmiş model ağırlıkları şirkete ait olduğundan bu depoda **paylaşılmamaktadır**.

## Geliştirici

**Emir** · Bilgisayar Mühendisliği, Balıkesir Üniversitesi

## Lisans

Lisans bilgisi buraya eklenecek.
