# Katkı Rehberi

EvrenAid'e katkı vermek istediğiniz için teşekkürler. Bu belge, projeyi
herkes için öngörülebilir tutan kuralları özetler.

## Hızlı başlangıç

```bash
git clone https://github.com/metabtw/evrenaideneme.git
cd evrenaideneme
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Repo `main` dalı üzerinden gelişir. Doğrudan `main`'e commit atmayın; kısa
ömürlü bir dal açın:

```bash
git switch -c feat/kisa-aciklama
```

## Commit kuralı — Conventional Commits

Biçim: `tip(kapsam): özet`

| Tip | Ne zaman |
| --- | --- |
| `feat` | Yeni özellik (M1, M2 …) |
| `fix` | Hata düzeltmesi |
| `refactor` | Davranış değişmeyen kod düzenlemesi |
| `perf` | Performans iyileştirmesi (FPS, gecikme) |
| `test` | Yalnızca test ekleme/düzeltme |
| `docs` | README, ROADMAP, docstring |
| `chore` | Bağımlılık, CI, yapılandırma |
| `build` | Paketleme, dağıtım |

**İyi:** `feat(detection): sınıf filtresi desteği ekle`
**Kötü:** `düzeltmeler`, `update`, `son hali`

- Özet satırı 72 karakteri geçmesin, sonunda nokta olmasın, emir kipi kullanın.
- Gövdede *neden* yaptığınızı anlatın; *ne* yaptığınız diff'te görünüyor.
- Bir commit tek bir mantıksal değişiklik olsun.

## Kod standartları

- **Python 3.10–3.12.** `from __future__ import annotations` kullanın.
- **Biçim/lint:** `ruff check app tests` ve `ruff format app tests` temiz olmalı
  (CI bunu zorunlu tutar).
- **Tip ipuçları:** Genel API'lerde (sınıf metotları, modül fonksiyonları) tam
  tip verin. `Any`'den kaçının; gerekiyorsa nedenini yorumda belirtin.
- **Docstring:** Türkçe, kısa ve *neden* odaklı. Kullanıcıya görünen her hata
  mesajı Türkçe ve eyleme dönük olmalı ("Kurulum: pip install -r requirements.txt").
- **Yorumlar:** Yalnızca kodun kendini anlatmadığı yerde (donanım tuhaflıkları,
  thread sınırları, performans tuzakları).
- **Satır uzunluğu:** 100 karakter (ruff yapılandırması).

## Mimari kurallar

Bunlar keyfi değil; performans bütçesini korumak için var:

1. **Ana döngü asla bloklanmaz.** Kamera, tespit, VLM ve TTS ayrı thread'lerde
   çalışır. Ana thread yalnızca okur, çizer, tuş işler.
2. **Kare paylaşımı kopyayla.** Bir kareyi başka thread'e verirken
   `frame.copy()` kullanın; referans paylaşmayın.
3. **Latest-frame-wins.** Kuyruklar `Queue(maxsize=1)` mantığıyla çalışır; eski
   kare birikmez, yenisi eskisini değiştirir.
4. **Ağır bağımlılıklar isteğe bağlı içe aktarılır.** `cv2`, `ultralytics`,
   `requests`, `pyttsx3` modül düzeyinde değil, ihtiyaç anında import edilir.
   Bu sayede `pytest` donanımsız ortamda çalışır.
5. **Test edilebilirlik için enjeksiyon.** Model, HTTP oturumu ve kamera
   fabrikası parametre olarak verilebilir (`model_factory`, `session`,
   `capture_factory`). Testlerde gerçek donanım/ağ kullanılmaz.

## Test

```bash
pytest                      # tümü
pytest -q tests/test_detector.py
pytest --cov=app --cov-report=term-missing
```

- Yeni davranış → yeni test. Hata düzeltmesi → hatayı yakalayan test.
- Donanım/ağ gerektiren testler `pytest.importorskip` veya sahte nesne ile
  yazılmalı; CI'da `torch`/`ultralytics` kurulmaz.
- Testler Türkçe adlandırılabilir; test *adı* davranışı anlatmalı.

## Sırlar ve gizlilik

- `.env`, API anahtarları, model ağırlıkları (`*.pt`), loglar ve kayıtlar
  **commit edilmez** (bkz. `.gitignore`).
- Kamera görüntüleri varsayılan olarak diske yazılmaz. Yeni bir özellik görüntü
  kaydedecekse açık kullanıcı onayı ve belgelenmiş bir ayar gerekir.
- Yanlışlıkla anahtar commit'lendiyse: anahtarı **iptal edin**, geçmişi
  temizleyin ve durumu bildirin.

## Pull request

1. Dalınızı `main` ile güncel tutun (`git rebase main`).
2. `pytest` ve `ruff` yeşil olsun.
3. PR açıklamasında: ne değişti, neden, nasıl doğruladınız (komut çıktısı),
   hangi ROADMAP maddesi kapanıyor.
4. Performansı etkileyen değişikliklerde önce/sonra ölçümü ekleyin
   (`--no-gui --duration 10` çıktısı yeterli).
5. İlgili ROADMAP kutularını işaretleyin; "Durum" satırını güncelleyin.

## Sorun bildirimi

Şunları ekleyin: işletim sistemi, Python sürümü, kamera indeksi, `python -m
app.main --list-cameras` çıktısı, ilgili log satırları (`logs/app.log`) ve
beklenen/gerçekleşen davranış. Ekran görüntüsü varsa HUD'u kırpmadan ekleyin.
