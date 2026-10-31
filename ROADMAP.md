# EvrenAid — Yol Haritası

> Hibrit YOLOv8 + Vision LLM masaüstü kamera asistanı.
> Bu dosya yaşayan bir plandır: her faz tamamlandıkça kutular işaretlenir ve
> "Durum" satırı güncellenir.

**Vizyon:** Kullanıcının kamerasını, 30 FPS akıcılığını kaybetmeden konuşan ve
açıklama yapan bir asistana dönüştürmek. Her şey yerelde, Türkçe ve düşük gecikmeyle.

**Kuzey yıldızı metrikleri**

1. Kamera akışı analiz sırasında **hiç donmaz** (0 düşen kare).
2. `s` tuşundan ilk sesli kelimeye kadar **< 5 sn** (GPU, 7B model).
3. Türkçe yanıtlar **anlaşılır ve kısa** (≤ 3 cümle, ≤ 256 token).
4. Kurulum **tek komutla** tekrarlanabilir (`pip install -r requirements.txt` + `ollama pull`).

---

## Faz özeti

| Faz | Ad | Çıktı | Durum |
| --- | --- | --- | --- |
| M0 | Temel iskelet | Depo, README, ROADMAP, config, CI | 🟢 Tamam (CONTRIBUTING hariç) |
| M1 | Canlı tespit | YOLOv8 + webcam + HUD, 30 FPS | 🟡 Kod hazır |
| M2 | Derin analiz | Ollama VLM istemcisi + Türkçe prompt | ⬜ |
| M3 | Sesli geri bildirim | pyttsx3/gTTS motorları | ⬜ |
| M4 | Orkestrasyon | Thread'ler, kuyruklar, dayanıklılık | ⬜ |
| M5 | Ürünleşme | UX, paketleme, ölçüm raporu | ⬜ |
| M6 | İleri özellikler | OCR modu, oto-tetik, kayıt | 💡 |

---

## M0 — Temel İskelet

**Hedef:** Kod yazmadan önce tekrarlanabilir bir zemin.

- [x] Depo oluşturma ve `main` dalı
- [x] `README.md` (mimari, kurulum, kullanım)
- [x] `ROADMAP.md` (bu dosya)
- [x] `.gitignore` (venv, `.env`, model ağırlıkları, loglar)
- [x] `.env.example` (sırsız yapılandırma şablonu)
- [x] `requirements.txt` (sabitlenmiş alt/üst sınırlar)
- [x] `LICENSE` (MIT)
- [x] `app/` paket iskeleti + boş modül dosyaları
- [x] `app/config/settings.py`: `.env` → dataclass, doğrulama, anlaşılır hata mesajları
- [x] `app/utils/logging.py`: konsol + döner dosya logu
- [x] `pytest` kurulumu ve ilk testler (`test_config.py`)
- [x] GitHub Actions: `ruff` + `pytest` (Python 3.11/3.12 matrisi)
- [ ] `CONTRIBUTING.md` ve commit kuralı (Conventional Commits)

**Kabul kriteri:** `git clone` → `pip install -r requirements.txt` → `pytest` yeşil;
`.env` yokken uygulama anlaşılır hata verir, çökmez.

**Riskler:** Python sürüm uyumsuzlukları (`pyttsx3`, `ultralytics`). → 3.11/3.12 CI matrisi.

---

## M1 — Canlı Tespit Katmanı

**Hedef:** Kameradan gelen görüntüde nesneleri gerçek zamanlı işaretlemek.

- [x] `camera/stream.py`: `VideoCapture` sarmalayıcı (kendi thread'i, kare kopyası)
  - Windows'ta `cv2.CAP_DSHOW`, macOS'ta `cv2.CAP_AVFOUNDATION`
  - Çözünürlük/FPS ayarı, açılamazsa net hata
  - `read()` başarısızlığında otomatik yeniden bağlanma (3 deneme)
- [x] `detection/detector.py`: YOLOv8 sarmalayıcı
  - Model tek seferlik yükleme (lazy singleton)
  - `conf`, `iou`, `imgsz`, `classes` filtreleri
  - `detect(frame) -> list[Detection]` (saf veri, çizim yok)
  - `detection/draw.py`: `draw_detections(frame, detections) -> frame` (saf görselleştirme)
- [x] `detection/worker.py`: tespit için ayrı thread + latest-frame-wins
- [x] `ui/overlay.py`: FPS sayacı, durum rozeti, Türkçe sınıf adları
- [x] `main.py`: yakalama döngüsü + tuş yönetimi (`s`, `t`, `q`)
- [x] `--no-gui` baş bayrağı (başsız test / CI için)
- [x] Testler: `detector` çıktısı sahte kare ile; sınıf adı eşlemesi
- [ ] Gerçek webcam ile kabul: 720p'de ≥ 25 FPS, `q` ile temiz çıkış

**Kabul kriteri:** 720p'de ≥ 25 FPS (GPU'suz dizüstü), kutular doğru konumda,
`q` ile temiz çıkış, kamera yokken anlaşılır hata.

**Durum notu (M1):** Kod yolu ve birim testleri tamam. Gerçek webcam ile
`python -m app.main --no-gui --duration 10` çalıştırıldı: YOLOv8n CPU'da
720p'de kare başına ~26–43 ms çıkarım, akış ~21 FPS (bu makinedeki kamera
sürücüsü 720p'de ~21 FPS veriyor; darboğaz yazılım değil donanım/sürücü).
Kutular ve Türkçe etiketler doğru çizildi (`kişi`, `sandalye`).

**Riskler:** Kamera indeksi/izin sorunları; düşük ışıkta düşük güven.
→ `APP_CONFIDENCE` ayarı ve `--list-cameras` yardımcısı.

---

## M2 — Derin Analiz Katmanı (Vision LLM)

**Hedef:** Tek kareyi yerel modele gönderip Türkçe, kullanılabilir yanıt almak.

- [ ] `analysis/vlm_client.py`
  - `OllamaClient`: `POST /api/generate`, `images: [base64]`, `stream: false`
  - `OpenAICompatibleClient`: `/v1/chat/completions` (opsiyonel bulut)
  - Ortak arayüz: `describe(frame_jpeg: bytes, prompt: str) -> str`
  - Zaman aşımı, yeniden deneme (2, üstel geri çekilme), anlamlı hata tipleri
- [ ] `utils/frames.py`: `encode_jpeg(frame, quality=85)`, boyut küçültme (max 1024 px)
- [ ] `analysis/prompts.py`: Türkçe şablonlar
  - `DESCRIBE`: "Sahneyi 3 cümlede Türkçe açıkla; nesneleri ve konumlarını belirt."
  - `OCR`: "Görseldeki tüm metni olduğu gibi yaz, yorum ekleme."
  - `COUNT`: "Şu nesneden kaç tane var? Sadece sayı ve kısa gerekçe."
  - Model kimliğine göre şablon seçimi (`llava` vs `qwen2.5vl`)
- [ ] Ollama yoksa/ model yoksa: kurulum yönlendirmesi (`ollama pull ...`)
- [ ] Testler: sahte HTTP sunucusu ile istemci sözleşmesi; prompt snapshot testi

**Kabul kriteri:** `python -m app.analyze --image ornek.jpg` çalışır; 7B model GPU'da
< 5 sn, CPU'da < 15 sn yanıt verir; Türkçe yanıt akıcı.

**Riskler:** Model VRAM'i aşarsa yavaşlar; `llava` Türkçede zayıf kalabilir.
→ Varsayılan `qwen2.5vl:7b`, alternatif `llava:7b`; model seçimi `.env`'de.

---

## M3 — Sesli Geri Bildirim

**Hedef:** Yanıtı doğal Türkçe ile seslendirmek, kesilebilir şekilde.

- [ ] `tts/base.py`: `SpeechEngine` protokolü (`speak`, `stop`, `is_busy`)
- [ ] `tts/pyttsx3_engine.py`: çevrimdışı, `tr` ses seçimi, hız/volume
- [ ] `tts/gtts_engine.py`: MP3 üretimi + oynatma, geçici dosya temizliği
- [ ] Metin normalizasyonu: sayılar, kısaltmalar, kod bloklarının atlanması
- [ ] Uzun yanıtları cümlelere bölüp sırayla okuma; `stop()` ile anında kesme
- [ ] TTS yoksa sessiz mod + ekranda uyarı (uygulama çökmez)
- [ ] Testler: metin bölme, normalizasyon; motor mock'u

**Kabul kriteri:** `s` sonrası yanıt ≤ 500 ms içinde konuşmaya başlar; `t` ile sessize
alınır; Türkçe telaffuz anlaşılır.

**Riskler:** Windows'ta `pywin32` eksikliği; Linux'ta `espeak-ng` yokluğu.
→ README kurulum notları + başlangıçta bağımlılık kontrolü.

---

## M4 — Asenkron Orkestrasyon ve Dayanıklılık

**Hedef:** Ağır işler arka planda; ana döngü asla beklemez.

- [ ] `state.py`: kilitli paylaşılan durum (son yanıt, durum makinesi, istatistikler)
- [ ] `analysis/worker.py`: tek worker thread + `queue.Queue(maxsize=1)`
  - **Single-flight:** istek sürerken yeni tetik → "meşgul" geri bildirimi
  - **Latest-frame-wins:** kuyrukta bekleyen eski kare varsa yenisiyle değiştir
- [ ] Durum makinesi: `IDLE → CAPTURING → ANALYZING → SPEAKING → IDLE`
- [ ] Kapanış protokolü: `threading.Event` ile temiz `join`, kaynak serbest bırakma
- [ ] Hata yönetimi: VLM/TTS hatası kullanıcıya Türkçe mesaj, döngü devam
- [ ] `app/bench.py`: FPS, gecikme, bellek ölçümü → `reports/bench-<tarih>.json`
- [ ] Stres testi: 60 sn boyunca her 2 sn tetikleme, sızıntı/donma kontrolü

**Kabul kriteri:** 60 sn stres testinde ana döngü ≥ 28 FPS ortalamada kalır; bellek
artışı < %10; çıkışta asılı thread yok.

**Riskler:** GIL ve OpenCV çizim çakışması; kuyruk taşması.
→ Kare kopyalama, kilit kapsamını dar tutma, `maxsize=1` + düşürme politikası.

---

## M5 — Ürünleşme

**Hedef:** Kurulabilir, anlaşılır, ölçülebilir bir uygulama.

- [ ] İlk açılış sihirbazı: kamera seçimi, model kontrolü, ses testi
- [ ] Türkçe hata mesajları sözlüğü
- [ ] `pyproject.toml` + `pipx install .` / `uv` desteği
- [ ] Windows için `PyInstaller` paketi (opsiyonel), macOS `.app`
- [ ] Ekran görüntüleri ve 30 sn'lik demo GIF (`assets/`)
- [ ] Performans raporu: CPU vs GPU, `llava` vs `qwen2.5vl` tablosu
- [ ] Sürüm etiketi `v0.1.0` ve changelog

**Kabul kriteri:** Yeni bir kullanıcı README'yi takip ederek 15 dakikada çalıştırır.

---

## M6 — İleri Özellikler (fikir havuzu)

- [ ] Otomatik tetikleme: belirli sınıf + güven eşiği + cooldown (ör. "kitap" 3 sn görülürse)
- [ ] Sürekli mod: her N saniyede sessiz analiz, yalnızca değişiklikte konuşma
- [ ] OCR odaklı mod: kırpma + `qwen2.5vl` ile yüksek doğruluklu metin okuma
- [ ] Konuşma geçmişi paneli ve dışa aktarma (JSON/Markdown)
- [ ] Oturum kaydı (video + yanıt zaman çizelgesi) — açık onay ile
- [ ] Çoklu kamera / ekran yakalama (`mss`) desteği
- [ ] Yerel STT ile soru sorma ("Bu ne?"), tam sesli döngü
- [ ] Web panel (FastAPI + WebSocket) ile uzaktan izleme
- [ ] Model önbelleği ve ilk yükleme süresini kısaltma (warm-up)

---

## Risk kaydı

| Risk | Etki | Olasılık | Azaltma |
| --- | --- | --- | --- |
| 7B model CPU'da çok yavaş | Yüksek | Orta | Küçük model (`moondream`), kare küçültme, `num_predict` sınırı |
| VRAM yetersizliği | Yüksek | Orta | `qwen2.5vl:3b`, CPU fallback, net hata mesajı |
| Türkçe yanıt kalitesi düşük | Orta | Orta | Türkçe prompt + few-shot, `qwen2.5vl` varsayılan |
| Kamera izni / indeks sorunu | Orta | Yüksek | `--list-cameras`, README sorun giderme |
| TTS bağımlılık sorunları | Orta | Orta | Motor soyutlaması, sessiz mod |
| Thread yarışı / donma | Yüksek | Düşük | Kilitli durum, tek worker, stres testi |
| Sırların repoya sızması | Yüksek | Düşük | `.gitignore`, `gitleaks` ön-kontrolü, `.env.example` |

---

## Her faz için Definition of Done

1. Kod `ruff` ve `black` denetiminden geçer.
2. Yeni davranış için test var; `pytest` yeşil.
3. README/ROADMAP ilgili bölümü güncellendi.
4. Türkçe hata mesajı ve log satırı mevcut.
5. Performans bütçesi ihlal edilmiyorsa ölçüm notu eklendi.

---

## Sürüm planı

| Sürüm | İçerik |
| --- | --- |
| `v0.1.0` | M0–M2: canlı tespit + tek kare analizi (konsol) |
| `v0.2.0` | M3: sesli geri bildirim |
| `v0.3.0` | M4: asenkron orkestrasyon + benchmark |
| `v1.0.0` | M5: paketlenmiş, belgelenmiş, ölçülmüş sürüm |
