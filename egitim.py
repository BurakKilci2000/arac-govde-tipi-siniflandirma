"""
egitim.py
=========
Araba govde tipi siniflandirma modeli - egitim ve degerlendirme scripti.

Mimari: EfficientNet-B0 (transfer ogrenme, ImageNet on-egitilmis)
  - Neden bu model?
      * Kucuk (~20 MB state_dict)        -> 95 MB sinirinin cok altinda
      * Compound scaling sayesinde       -> kucuk ama yuksek dogruluk
      * ImageNet on-egitimi ile          -> az veriyle iyi genelleme
      * Hizli cikarim (inference)        -> arayuzde bekleme yok
  - Son siniflandirici katmani 8 sinifa gore degistirilmistir.

Ciktilar:
    ciktilar/
        en_iyi_model.pth            <- en iyi dogrulama_dogrulugu'na sahip agirliklar
        egitim_gecmisi.csv          <- her epoch icin kayit
        kayip_grafigi.png           <- Egitim & Dogrulama Loss
        dogruluk_grafigi.png        <- Egitim & Dogrulama Accuracy
        karisiklik_matrisi.png      <- Normalize edilmis 8x8 matris
        siniflandirma_raporu.txt    <- precision/recall/f1 (sinif basi & ortalama)

Kullanim:
    python egitim.py --veri_klasor veri --epoch 30 --batch_boyut 32
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm


# ImageFolder klasorleri alfabetik sirayla okudugu icin
# bizim sinif sirasi da alfabetik olmali. Bu listeyi her yerde
# (egitim, tahmin, arayuz) sabit tutuyoruz.
SINIF_ISIMLERI = [
    "ACIK_TEKERLEKLI",
    "HATCHBACK",
    "MICRO",
    "PICK_UP",
    "SEDAN",
    "STATION_WAGON",
    "SUV",
    "VAN",
]
SINIF_SAYISI = len(SINIF_ISIMLERI)
GORSEL_BOYUTU = 224  # EfficientNet-B0'in dogal giris boyutu


# ---------------------------------------------------------------------------
# 1. VERI ON ISLEMLERI (preprocessing + augmentation)
# ---------------------------------------------------------------------------
def donusumleri_olustur():
    """
    Egitim seti icin agresif veri cogaltma, dogrulama/test icin sadece
    yeniden boyutlandir + normalize.

    ImageNet ortalama/std degerleri kullaniliyor cunku ImageNet on-egitimli
    omurga (backbone) bu istatistikleri bekler. Aksi halde "input distribution
    shift" yasanir ve model dogru ogrenmemis gibi davranir.
    """
    imagenet_ortalama = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    egitim_donusumu = transforms.Compose([
        # Rastgele kirpma + yeniden boyutlandirma: konum/olcek invaryansi
        transforms.RandomResizedCrop(GORSEL_BOYUTU, scale=(0.7, 1.0)),
        # Yatay cevirme: arabalar icin dogal bir cogaltma
        transforms.RandomHorizontalFlip(p=0.5),
        # Hafif rotasyon: kameranin egri tutuldugu durumlar
        transforms.RandomRotation(degrees=15),
        # Renk varyasyonu: farkli isik kosullarina dayaniklilik
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        # Bazen perspektif: farkli acilardan cekilmis gibi davransin
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_ortalama, std=imagenet_std),
        # Random erasing: arka plan parcalarini silerek modele
        # objenin tamamina bakmayi ogretir (overfitting'i azaltir).
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    degerlendirme_donusumu = transforms.Compose([
        transforms.Resize(256),                # once kisa kenardan 256
        transforms.CenterCrop(GORSEL_BOYUTU),  # sonra merkezden 224 kirp
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_ortalama, std=imagenet_std),
    ])

    return egitim_donusumu, degerlendirme_donusumu


# ---------------------------------------------------------------------------
# 2. MODEL TANIMI
# ---------------------------------------------------------------------------
def modeli_olustur(sinif_sayisi: int = SINIF_SAYISI,
                   dropout: float = 0.3) -> nn.Module:
    """
    ImageNet on-egitimli EfficientNet-B0'i yukler ve siniflandiricisini
    8 sinif icin degistirir.

    Stratejim:
      - Omurgayi (backbone) DONDURMUYORUM; tum katmanlari fine-tune ediyorum.
      - Ancak diferansiyel ogrenme orani kullaniyorum:
        omurga icin kucuk lr, yeni siniflandirici basligi icin buyuk lr.
      - Orijinal siniflandirici: Dropout(0.2) + Linear(1280, 1000)
        Yeni baslik:             Dropout(p)   + Linear(1280, 8)
    """
    agirliklar = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=agirliklar)

    # Siniflandiricinin giris boyutunu ogren (1280 oldugunu biliyorum,
    # ama kod dinamik kalsin diye nesnesinden okuyorum)
    giris_ozellik_sayisi = model.classifier[1].in_features

    # 8 sinifa uygun yeni baslik
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(giris_ozellik_sayisi, sinif_sayisi),
    )
    return model


# ---------------------------------------------------------------------------
# 3. EGITIM / DOGRULAMA DONGULERI
# ---------------------------------------------------------------------------
def bir_epoch_calistir(model, yukleyici, kayip_fonksiyonu, optimize_edici,
                       cihaz, egitim: bool):
    """
    Tek bir epoch calistirir.
      - egitim=True ise gradyan gunceller (geri yayilim yapilir)
      - egitim=False ise sadece tahmin uretir (dogrulama/test icin)
    """
    if egitim:
        model.train()
    else:
        model.eval()

    toplam_kayip = 0.0
    dogru_sayisi = 0
    toplam_ornek = 0

    # torch.set_grad_enabled: egitim ise gradyan acik, dogrulama'da kapali.
    # Kapali olmasi GPU bellegini ve hizi onemli olcude iyilestirir.
    with torch.set_grad_enabled(egitim):
        durum_cubugu = tqdm(yukleyici,
                            desc="egitim   " if egitim else "dogrulama",
                            leave=False)
        for gorseller, etiketler in durum_cubugu:
            gorseller = gorseller.to(cihaz, non_blocking=True)
            etiketler = etiketler.to(cihaz, non_blocking=True)

            ciktilar = model(gorseller)
            kayip = kayip_fonksiyonu(ciktilar, etiketler)

            if egitim:
                optimize_edici.zero_grad()
                kayip.backward()
                optimize_edici.step()

            # Batch ici istatistikleri biriktir
            toplam_kayip += kayip.item() * gorseller.size(0)
            tahminler = ciktilar.argmax(dim=1)
            dogru_sayisi += (tahminler == etiketler).sum().item()
            toplam_ornek += etiketler.size(0)

            durum_cubugu.set_postfix(
                kayip=f"{kayip.item():.3f}",
                dogruluk=f"{dogru_sayisi / toplam_ornek:.3f}")

    epoch_kayip = toplam_kayip / toplam_ornek
    epoch_dogruluk = dogru_sayisi / toplam_ornek
    return epoch_kayip, epoch_dogruluk


# ---------------------------------------------------------------------------
# 4. TEST / DEGERLENDIRME (tek gecisle tum tahminleri toplar)
# ---------------------------------------------------------------------------
@torch.no_grad()
def degerlendir(model, yukleyici, cihaz):
    """Tum tahminleri ve gercek etiketleri dondurur (metrik hesabi icin)."""
    model.eval()
    tum_tahminler, tum_etiketler = [], []
    for gorseller, etiketler in tqdm(yukleyici, desc="test", leave=False):
        gorseller = gorseller.to(cihaz, non_blocking=True)
        ciktilar = model(gorseller)
        tahminler = ciktilar.argmax(dim=1).cpu().numpy()
        tum_tahminler.extend(tahminler)
        tum_etiketler.extend(etiketler.numpy())
    return np.array(tum_etiketler), np.array(tum_tahminler)


# ---------------------------------------------------------------------------
# 5. GORSELLESTIRMELER (proje sarti: 3 grafik)
# ---------------------------------------------------------------------------
def egrileri_ciz(gecmis: dict, cikti_klasor: Path):
    """Loss ve Accuracy egrilerini PNG olarak kaydeder."""
    epoch_listesi = range(1, len(gecmis["egitim_kayip"]) + 1)

    # --- Grafik 1: Egitim & Dogrulama Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_listesi, gecmis["egitim_kayip"],
             "-o", label="Egitim Loss")
    plt.plot(epoch_listesi, gecmis["dogrulama_kayip"],
             "-s", label="Dogrulama Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Egitim & Dogrulama Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(cikti_klasor / "kayip_grafigi.png", dpi=150)
    plt.close()

    # --- Grafik 2: Egitim & Dogrulama Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_listesi, [d * 100 for d in gecmis["egitim_dogruluk"]],
             "-o", label="Egitim Accuracy")
    plt.plot(epoch_listesi, [d * 100 for d in gecmis["dogrulama_dogruluk"]],
             "-s", label="Dogrulama Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Egitim & Dogrulama Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(cikti_klasor / "dogruluk_grafigi.png", dpi=150)
    plt.close()


def karisiklik_matrisini_ciz(gercek_etiketler, tahminler,
                             sinif_isimleri, cikti_klasor: Path):
    """Normalize edilmis confusion matrix (her satir kendi icinde toplam=1)."""
    matris = confusion_matrix(gercek_etiketler, tahminler,
                              labels=list(range(len(sinif_isimleri))))
    # Satir bazinda normalize et: her gercek sinifin tahmin dagilimi
    matris_norm = matris.astype(float) / matris.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(9, 7))
    sns.heatmap(matris_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=sinif_isimleri, yticklabels=sinif_isimleri,
                cbar_kws={"label": "Oran"})
    plt.xlabel("Tahmin Edilen Sinif")
    plt.ylabel("Gercek Sinif")
    plt.title("Normalize Edilmis Karisiklik Matrisi")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cikti_klasor / "karisiklik_matrisi.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 6. ANA AKIS
# ---------------------------------------------------------------------------
def ana():
    ayrastirici = argparse.ArgumentParser()
    ayrastirici.add_argument("--veri_klasor", type=str, default="veri")
    ayrastirici.add_argument("--cikti_klasor", type=str, default="ciktilar")
    ayrastirici.add_argument("--epoch", type=int, default=30)
    ayrastirici.add_argument("--batch_boyut", type=int, default=32)
    ayrastirici.add_argument("--ogrenme_orani", type=float, default=1e-3,
                             help="siniflandirici baslik icin learning rate")
    ayrastirici.add_argument("--ogrenme_orani_omurga", type=float, default=1e-4,
                             help="on-egitimli omurga icin DAHA DUSUK lr "
                                  "(ogrenilmis ozellikleri bozmamak icin)")
    ayrastirici.add_argument("--agirlik_zayiflama", type=float, default=1e-4,
                             help="weight decay - L2 regulasyonu")
    ayrastirici.add_argument("--dropout", type=float, default=0.3)
    ayrastirici.add_argument("--sabir", type=int, default=7,
                             help="Erken durdurma sabri: dogrulama_kayip "
                                  "kac epoch dusmezse durulsun")
    ayrastirici.add_argument("--isci_sayisi", type=int, default=4)
    ayrastirici.add_argument("--tohum", type=int, default=42)
    parametreler = ayrastirici.parse_args()

    # Tekrarlanabilirlik
    torch.manual_seed(parametreler.tohum)
    np.random.seed(parametreler.tohum)

    cikti_klasor = Path(parametreler.cikti_klasor)
    cikti_klasor.mkdir(parents=True, exist_ok=True)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {cihaz}")

    # --- Veri yukleyiciler ---
    egitim_donusumu, degerlendirme_donusumu = donusumleri_olustur()
    veri_klasor = Path(parametreler.veri_klasor)

    egitim_seti = datasets.ImageFolder(veri_klasor / "egitim",
                                       transform=egitim_donusumu)
    dogrulama_seti = datasets.ImageFolder(veri_klasor / "dogrulama",
                                          transform=degerlendirme_donusumu)
    test_seti = datasets.ImageFolder(veri_klasor / "test",
                                     transform=degerlendirme_donusumu)

    # ImageFolder'in buldugu sinif sirasi bizim listemizle eslesmeli
    print("ImageFolder siniflari:", egitim_seti.classes)
    assert egitim_seti.classes == SINIF_ISIMLERI, (
        f"Sinif sirasi eslemiyor. Beklenen {SINIF_ISIMLERI}, "
        f"bulunan {egitim_seti.classes}"
    )

    egitim_yukleyici = DataLoader(egitim_seti,
                                  batch_size=parametreler.batch_boyut,
                                  shuffle=True,
                                  num_workers=parametreler.isci_sayisi,
                                  pin_memory=True)
    dogrulama_yukleyici = DataLoader(dogrulama_seti,
                                     batch_size=parametreler.batch_boyut,
                                     shuffle=False,
                                     num_workers=parametreler.isci_sayisi,
                                     pin_memory=True)
    test_yukleyici = DataLoader(test_seti,
                                batch_size=parametreler.batch_boyut,
                                shuffle=False,
                                num_workers=parametreler.isci_sayisi,
                                pin_memory=True)

    print(f"Egitim: {len(egitim_seti)}  "
          f"Dogrulama: {len(dogrulama_seti)}  "
          f"Test: {len(test_seti)}")

    # --- Model ---
    model = modeli_olustur(sinif_sayisi=SINIF_SAYISI,
                           dropout=parametreler.dropout).to(cihaz)

    # --- Kayip Fonksiyonu & Optimize Edici ---
    # Sinif dengesizligine karsi: her sinifin egitim setindeki frekansinin
    # tersini agirlik olarak kullanirim. Tamamen dengeliyse etkisi minimumdur.
    sinif_sayilari = np.bincount(
        [y for _, y in egitim_seti.samples], minlength=SINIF_SAYISI)
    sinif_agirliklari = torch.tensor(
        len(egitim_seti) / (SINIF_SAYISI * sinif_sayilari.clip(min=1)),
        dtype=torch.float32, device=cihaz,
    )
    print("Sinif agirliklari:", sinif_agirliklari.cpu().numpy().round(2))

    # Label smoothing (etiket yumusatma) -> asiri guvenli tahminleri azaltir,
    # genelleme kabiliyetini artirir.
    kayip_fonksiyonu = nn.CrossEntropyLoss(weight=sinif_agirliklari,
                                           label_smoothing=0.1)

    # Diferansiyel learning rate: omurga kucuk, baslik buyuk
    omurga_parametreleri = [p for n, p in model.named_parameters()
                            if not n.startswith("classifier")]
    baslik_parametreleri = [p for n, p in model.named_parameters()
                            if n.startswith("classifier")]
    optimize_edici = optim.AdamW([
        {"params": omurga_parametreleri,
         "lr": parametreler.ogrenme_orani_omurga},
        {"params": baslik_parametreleri,
         "lr": parametreler.ogrenme_orani},
    ], weight_decay=parametreler.agirlik_zayiflama)

    # Cosine annealing -> sona dogru ogrenme oranini yumusakca dusurur,
    # daha kararli bir minimuma oturur.
    planlayici = optim.lr_scheduler.CosineAnnealingLR(
        optimize_edici, T_max=parametreler.epoch, eta_min=1e-6
    )

    # --- Egitim dongusu + erken durdurma ---
    gecmis = {"egitim_kayip": [], "egitim_dogruluk": [],
              "dogrulama_kayip": [], "dogrulama_dogruluk": []}
    en_iyi_dogrulama_kaybi = float("inf")
    iyilesmeyen_epoch_sayisi = 0
    en_iyi_model_yolu = cikti_klasor / "en_iyi_model.pth"

    for epoch_no in range(1, parametreler.epoch + 1):
        baslik_lr = optimize_edici.param_groups[1]['lr']
        print(f"\nEpoch {epoch_no}/{parametreler.epoch}  "
              f"(baslik_lr={baslik_lr:.2e})")

        e_kayip, e_dogruluk = bir_epoch_calistir(
            model, egitim_yukleyici, kayip_fonksiyonu,
            optimize_edici, cihaz, egitim=True)
        d_kayip, d_dogruluk = bir_epoch_calistir(
            model, dogrulama_yukleyici, kayip_fonksiyonu,
            optimize_edici, cihaz, egitim=False)
        planlayici.step()

        gecmis["egitim_kayip"].append(e_kayip)
        gecmis["egitim_dogruluk"].append(e_dogruluk)
        gecmis["dogrulama_kayip"].append(d_kayip)
        gecmis["dogrulama_dogruluk"].append(d_dogruluk)

        print(f"  egitim:    kayip={e_kayip:.4f} dogruluk={e_dogruluk:.4f}")
        print(f"  dogrulama: kayip={d_kayip:.4f} dogruluk={d_dogruluk:.4f}")

        # En iyi modeli kaydet (dogrulama kaybi azaliyorsa)
        if d_kayip < en_iyi_dogrulama_kaybi - 1e-4:
            en_iyi_dogrulama_kaybi = d_kayip
            iyilesmeyen_epoch_sayisi = 0
            # state_dict + sinif isimleri + giris boyutu hep birlikte saklaniyor
            torch.save({
                "state_dict": model.state_dict(),
                "sinif_isimleri": SINIF_ISIMLERI,
                "gorsel_boyutu": GORSEL_BOYUTU,
                "mimari": "efficientnet_b0",
                "epoch": epoch_no,
                "dogrulama_kayip": d_kayip,
                "dogrulama_dogruluk": d_dogruluk,
            }, en_iyi_model_yolu)
            print(f"  >> En iyi model kaydedildi (kayip={d_kayip:.4f})")
        else:
            iyilesmeyen_epoch_sayisi += 1
            if iyilesmeyen_epoch_sayisi >= parametreler.sabir:
                print(f"\nErken durdurma: {parametreler.sabir} epoch'tur "
                      f"dogrulama kaybi azalmiyor.")
                break

    # --- Gecmisi CSV'ye yaz ---
    with open(cikti_klasor / "egitim_gecmisi.csv", "w", newline="") as f:
        yazici = csv.writer(f)
        yazici.writerow(["epoch", "egitim_kayip", "egitim_dogruluk",
                         "dogrulama_kayip", "dogrulama_dogruluk"])
        for i in range(len(gecmis["egitim_kayip"])):
            yazici.writerow([i + 1,
                             gecmis["egitim_kayip"][i],
                             gecmis["egitim_dogruluk"][i],
                             gecmis["dogrulama_kayip"][i],
                             gecmis["dogrulama_dogruluk"][i]])

    egrileri_ciz(gecmis, cikti_klasor)
    print(f"\nLoss & Accuracy grafikleri kaydedildi: {cikti_klasor}")

    # --- En iyi modeli yukleyip TEST setinde degerlendir ---
    print("\nTest seti uzerinde son degerlendirme yapiliyor...")
    kontrol_noktasi = torch.load(en_iyi_model_yolu,
                                 map_location=cihaz, weights_only=False)
    model.load_state_dict(kontrol_noktasi["state_dict"])
    gercek_etiketler, tahminler = degerlendir(model, test_yukleyici, cihaz)

    # F1 birinci oncelikli metrik
    makro_f1 = f1_score(gercek_etiketler, tahminler, average="macro")
    agirlikli_f1 = f1_score(gercek_etiketler, tahminler, average="weighted")

    rapor_metni = classification_report(
        gercek_etiketler, tahminler,
        target_names=SINIF_ISIMLERI, digits=4)

    print("\n" + rapor_metni)
    print(f"Makro F1     : {makro_f1:.4f}")
    print(f"Agirlikli F1 : {agirlikli_f1:.4f}")

    with open(cikti_klasor / "siniflandirma_raporu.txt", "w") as f:
        f.write(rapor_metni + "\n")
        f.write(f"Makro F1     : {makro_f1:.4f}\n")
        f.write(f"Agirlikli F1 : {agirlikli_f1:.4f}\n")

    karisiklik_matrisini_ciz(gercek_etiketler, tahminler,
                             SINIF_ISIMLERI, cikti_klasor)
    print(f"Karisiklik matrisi kaydedildi: "
          f"{cikti_klasor / 'karisiklik_matrisi.png'}")

    # Ozet meta bilgisi
    with open(cikti_klasor / "egitim_ozeti.json", "w") as f:
        json.dump({
            "calistirilan_epoch_sayisi": len(gecmis["egitim_kayip"]),
            "en_iyi_dogrulama_kaybi": en_iyi_dogrulama_kaybi,
            "test_makro_f1": makro_f1,
            "test_agirlikli_f1": agirlikli_f1,
            "sinif_isimleri": SINIF_ISIMLERI,
            "parametreler": vars(parametreler),
        }, f, indent=2, ensure_ascii=False)

    print("\nTum ciktilar hazir.")


if __name__ == "__main__":
    ana()
