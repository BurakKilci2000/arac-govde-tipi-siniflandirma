"""
veri_seti_hazirla.py
====================
Topladigim ham gorselleri 8 sinif klasoru altina yerlestirdikten sonra,
bu script onlari egitim/dogrulama/test alt klasorlerine otomatik olarak boler.

Beklenen baslangic yapisi:
    ham_veri/
        SUV/             <- buraya 200+ SUV gorseli
        VAN/
        STATION_WAGON/
        MICRO/
        ACIK_TEKERLEKLI/
        SEDAN/
        HATCHBACK/
        PICK_UP/

Cikti yapisi:
    veri/
        egitim/        (~%75)
        dogrulama/     (~%15)
        test/          (~%10)   <- kendi ic testimiz; final test sunumda
                                   hocalar tarafindan verilecek

Kullanim:
    python veri_seti_hazirla.py --ham_klasor ham_veri --cikti_klasor veri
"""

import argparse
import random
import shutil
from pathlib import Path

# 8 sinifin resmi adlari - proje dokumanindaki sira ile.
# Etiket sirasinin her yerde tutarli olmasi icin bu listeyi sabit tutuyorum.
SINIFLAR = [
    "SUV",
    "VAN",
    "STATION_WAGON",
    "MICRO",
    "ACIK_TEKERLEKLI",   # Acik tekerlekli (F1) araclari
    "SEDAN",
    "HATCHBACK",
    "PICK_UP",
]

# Destekledigim gorsel uzantilari (buyuk/kucuk harf duyarsiz)
GORSEL_UZANTILARI = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sinifi_bol(sinif_klasoru: Path, cikti_kok: Path, sinif_adi: str,
               dogrulama_orani: float, test_orani: float, tohum: int = 42):
    """Tek bir sinifin gorsellerini egitim/dogrulama/test'e dagitir."""
    # Tum gecerli gorselleri topla
    gorseller = [p for p in sinif_klasoru.iterdir()
                 if p.is_file() and p.suffix.lower() in GORSEL_UZANTILARI]

    if len(gorseller) == 0:
        print(f"[UYARI] {sinif_adi} sinifi bos, atlaniyor.")
        return

    # Tekrarlanabilir bolme icin sabit tohum
    random.seed(tohum)
    random.shuffle(gorseller)

    toplam = len(gorseller)
    n_test = int(toplam * test_orani)
    n_dogrulama = int(toplam * dogrulama_orani)
    n_egitim = toplam - n_dogrulama - n_test

    # Dilimle: ilk n_egitim -> egitim, sonraki n_dogrulama -> dogrulama,
    # geriye kalan -> test
    bolumler = {
        "egitim":    gorseller[:n_egitim],
        "dogrulama": gorseller[n_egitim:n_egitim + n_dogrulama],
        "test":      gorseller[n_egitim + n_dogrulama:],
    }

    for bolum_adi, dosyalar in bolumler.items():
        hedef_klasor = cikti_kok / bolum_adi / sinif_adi
        hedef_klasor.mkdir(parents=True, exist_ok=True)
        for kaynak in dosyalar:
            shutil.copy2(kaynak, hedef_klasor / kaynak.name)

    print(f"  {sinif_adi:18s} | toplam={toplam:4d} | "
          f"egitim={n_egitim:4d}  dogrulama={n_dogrulama:4d}  test={n_test:4d}")


def ana():
    """Ana akis: argumanlari oku, her sinifi bol, ozet yazdir."""
    ayrastirici = argparse.ArgumentParser()
    ayrastirici.add_argument("--ham_klasor", type=str, default="ham_veri",
                             help="8 sinif alt klasorunun bulundugu kok klasor")
    ayrastirici.add_argument("--cikti_klasor", type=str, default="veri",
                             help="Bolunmus veri setinin yazilacagi klasor")
    ayrastirici.add_argument("--dogrulama_orani", type=float, default=0.15)
    ayrastirici.add_argument("--test_orani", type=float, default=0.10)
    ayrastirici.add_argument("--tohum", type=int, default=42)
    parametreler = ayrastirici.parse_args()

    ham_kok = Path(parametreler.ham_klasor)
    cikti_kok = Path(parametreler.cikti_klasor)

    if not ham_kok.exists():
        raise FileNotFoundError(f"{ham_kok} bulunamadi.")

    # Daha once bolunmusse uzerine yazmadan once uyari ver
    if cikti_kok.exists():
        print(f"[UYARI] {cikti_kok} zaten var. Icindekiler birlestirilecek.")

    print(f"\nVeri seti bolunuyor: {ham_kok} -> {cikti_kok}")
    egitim_orani = 1 - parametreler.dogrulama_orani - parametreler.test_orani
    print(f"Oranlar: egitim={egitim_orani:.2f} "
          f"dogrulama={parametreler.dogrulama_orani} "
          f"test={parametreler.test_orani}\n")

    for sinif in SINIFLAR:
        sinif_klasoru = ham_kok / sinif
        if not sinif_klasoru.exists():
            print(f"[UYARI] {sinif} sinif klasoru yok, atlaniyor.")
            continue
        sinifi_bol(sinif_klasoru, cikti_kok, sinif,
                   parametreler.dogrulama_orani,
                   parametreler.test_orani,
                   parametreler.tohum)

    print("\nTamamlandi. Yeni yapi:")
    for bolum in ["egitim", "dogrulama", "test"]:
        bolum_klasor = cikti_kok / bolum
        if bolum_klasor.exists():
            toplam = sum(1 for p in bolum_klasor.rglob("*")
                         if p.is_file() and p.suffix.lower() in GORSEL_UZANTILARI)
            print(f"  {bolum}/  -> {toplam} gorsel")


if __name__ == "__main__":
    ana()
