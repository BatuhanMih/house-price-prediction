# House Price Prediction | Konut Fiyat Tahmini

[🇬🇧 English](#english) · [🇹🇷 Türkçe](#türkçe)

---

<a id="english"></a>

## English

This project compares an XGBoost regressor and a multilayer perceptron (MLP) for estimating residential sale prices from tabular property features. It was developed as a three-person student project.

The repository contains training scripts and data-loading tests. The original Excel dataset is not included in this public version because its collection source and redistribution rights have not been documented. The scripts can be run with a dataset you are permitted to use.

### Models and Evaluation

- **XGBoost:** An 80/20 train/test split, with five-fold cross-validation on the training portion before fitting the final model.
- **MLP:** The same default train/test split, with a validation subset drawn from the training portion for early stopping. Numeric inputs and the target are scaled using training data only.
- Both scripts report MAE, RMSE and R² on their held-out test portion and save predictions locally under `artifacts/`.

The default split seed is `24`. The two scripts use different model-selection procedures, so their outputs should not be interpreted as a controlled comparison of model families. Earlier project screenshots are not a reproducible benchmark for these scripts and are not included here.

### Input Data

Provide an `.xlsx` file with these columns:

| Column | Meaning |
|---|---|
| `Konum` | District or location |
| `Metrekare` | Floor area in square metres |
| `Oda Sayısı` | Room configuration |
| `Bina Yaşı` | Building age |
| `Kat Durumu` | Floor number or status |
| `Asansör` | Elevator availability |
| `Doğal Gaz` | Natural-gas availability |
| `Fiyat (TL)` | Sale price in Turkish lira |

The loader also recognizes a few damaged Turkish labels found in the archived workbook. It checks the required columns but does not establish the reliability or legal status of a new dataset. Do not upload personal or third-party listing data without permission.

### Run Locally

Use Python 3.10 or 3.11 and create a virtual environment. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Train XGBoost:

```bash
python -m src.train_xgboost --data path/to/your_data.xlsx
```

Train the MLP:

```bash
python -m src.train_mlp --data path/to/your_data.xlsx
```

Use `--help` on either command for the available settings. The default input path, if `--data` is omitted, is `data/emlak_fiyat.xlsx`. That file is deliberately ignored by Git.

Run the lightweight loader tests:

```bash
python -m pip install -r requirements-dev.txt
pytest
ruff check .
```

Training creates model and prediction files in `artifacts/`; those files are also ignored by Git.

### Limitations

The archived project used 500 rows and included 11 exact duplicate rows. Both a small sample and duplicate records can make holdout estimates unstable or optimistic. No model here should be used as a production appraisal system without a documented dataset, stronger validation and domain review.

### Team

- [Batuhan Mıh](https://github.com/BatuhanMih)
- [Mustafa Kurtulan](https://github.com/MustafaKurtulan)
- [Faruk Topal](https://github.com/faruktopal)

### Reuse

No software license has been granted for this repository. Contact the project contributors before reusing or redistributing the code. A license can be added after all contributors agree to its terms.

---

<a id="türkçe"></a>

## Türkçe

Bu proje, tablo biçimindeki konut özelliklerini kullanarak satış fiyatlarını tahmin etmek amacıyla XGBoost regresyon modeli ile çok katmanlı algılayıcı (MLP) modelini karşılaştırır. Üç kişilik bir öğrenci projesi olarak geliştirilmiştir.

Depoda eğitim betikleri ve veri yükleme testleri bulunmaktadır. Veri setinin toplandığı kaynak ve yeniden dağıtım hakları belgelenmediği için orijinal Excel veri seti bu herkese açık sürüme dahil edilmemiştir. Betikler, kullanımına izin verilen bir veri setiyle çalıştırılabilir.

### Modeller ve Değerlendirme

- **XGBoost:** Veriler %80 eğitim ve %20 test olacak şekilde ayrılır. Son model eğitilmeden önce eğitim verisi üzerinde beş katlı çapraz doğrulama uygulanır.
- **MLP:** Aynı varsayılan eğitim/test veri ayrımı kullanılır. Erken durdurma (early stopping) için eğitim verisinin içinden bir doğrulama alt kümesi ayrılır. Sayısal girdiler ve hedef değişken, yalnızca eğitim verisi kullanılarak ölçeklendirilir.
- Her iki betik de ayrılmış test verisi üzerinde MAE, RMSE ve R² metriklerini hesaplar. Tahmin sonuçları yerel olarak `artifacts/` dizinine kaydedilir.

Varsayılan veri ayrımı tohumu `24` olarak belirlenmiştir. İki betik farklı model seçimi yöntemleri kullandığından, elde edilen sonuçlar model aileleri arasında kontrollü bir karşılaştırma olarak değerlendirilmemelidir. Projenin önceki ekran görüntüleri, bu betiklerle yeniden üretilebilir bir kıyaslama sonucu değildir ve depoya dahil edilmemiştir.

### Girdi Verileri

Aşağıdaki sütunları içeren bir `.xlsx` dosyası sağlayın:

| Sütun | Açıklama |
|---|---|
| `Konum` | İlçe veya konum bilgisi |
| `Metrekare` | Metrekare cinsinden kullanım alanı |
| `Oda Sayısı` | Oda düzeni veya sayısı |
| `Bina Yaşı` | Binanın yaşı |
| `Kat Durumu` | Kat numarası veya kat bilgisi |
| `Asansör` | Asansör bulunup bulunmadığı |
| `Doğal Gaz` | Doğal gaz bulunup bulunmadığı |
| `Fiyat (TL)` | Türk lirası cinsinden satış fiyatı |

Veri yükleyici, arşivlenmiş çalışma kitabında bulunan bazı bozulmuş Türkçe sütun adlarını da tanır. Gerekli sütunların mevcut olup olmadığını kontrol eder; ancak yeni bir veri setinin güvenilirliğini veya hukuki kullanım durumunu doğrulamaz. İzin almadan kişisel verileri veya üçüncü taraflara ait ilan verilerini yüklemeyin.

### Yerel Ortamda Çalıştırma

Python 3.10 veya 3.11 kullanarak bir sanal ortam oluşturun. Bağımlılıkları yükleyin:

```bash
python -m pip install -r requirements.txt
```

XGBoost modelini eğitmek için:

```bash
python -m src.train_xgboost --data path/to/your_data.xlsx
```

MLP modelini eğitmek için:

```bash
python -m src.train_mlp --data path/to/your_data.xlsx
```

Kullanılabilir ayarları görüntülemek için her iki komuta da `--help` parametresini ekleyebilirsiniz. `--data` parametresi belirtilmezse varsayılan girdi yolu `data/emlak_fiyat.xlsx` olur. Bu dosya, Git tarafından izlenmeyecek şekilde yapılandırılmıştır.

Veri yükleyicinin temel testlerini çalıştırmak için:

```bash
python -m pip install -r requirements-dev.txt
pytest
ruff check .
```

Eğitim sonucunda model ve tahmin dosyaları `artifacts/` dizininde oluşturulur. Bu dosyalar da Git tarafından izlenmeyecek şekilde yapılandırılmıştır.

### Sınırlamalar

Arşivlenmiş projede 500 satır bulunmakta ve bunların 11'i tamamen aynı olan yinelenmiş kayıtlardan oluşmaktadır. Hem veri setinin küçük olması hem de yinelenen kayıtlar, test sonuçlarının kararsız veya gerçekte olduğundan daha başarılı görünmesine neden olabilir.

Belgelenmiş bir veri seti, daha güçlü doğrulama yöntemleri ve alan uzmanlarının değerlendirmesi olmadan bu modellerden hiçbiri gerçek bir konut değerleme sistemi olarak kullanılmamalıdır.

### Ekip

- [Batuhan Mıh](https://github.com/BatuhanMih)
- [Mustafa Kurtulan](https://github.com/MustafaKurtulan)
- [Faruk Topal](https://github.com/faruktopal)

### Yeniden Kullanım

Bu depo için herhangi bir yazılım lisansı verilmemiştir. Kodu yeniden kullanmadan veya dağıtmadan önce proje katkıda bulunanlarıyla iletişime geçin. Tüm katkıda bulunanlar lisans koşulları üzerinde anlaşmaya vardıktan sonra depoya uygun bir lisans eklenebilir.
