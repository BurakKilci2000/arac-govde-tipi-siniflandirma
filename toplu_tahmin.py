"""
toplu_tahmin.py
===============
Hocalarin paylastigi test scripti (Test.txt) ile UYUMLU cikti ureten script.

Bir klasordeki tum goruntuleri modele verir ve sonuclari
'preds.txt' dosyasina su formatta yazar:

    dosya_adi.jpg | label: SAYI

Sinif numaralari proje dokumanindaki sira ile verilir:
    1 = SUV
    2 = VAN
    3 = STATION WAGON
    4 = MICRO
    5 = ACIK TEKERLEKLI (F1)
    6 = SEDAN
    7 = HATCHBACK
    8 = PICK UP

Kullanim:
    python toplu_tahmin.py --gorsel_klasor test_gorselleri --model ciktilar/en_iyi_model.pth

    (--gorsel_klasor: hocalarin verecegi test goruntulerinin bulundugu klasor)
"""

import argparse
from pathlib import Path

from tahmin import Tahminleyici


# ---------------------------------------------------------------------------
# SINIF ESLESTIRME
# ---------------------------------------------------------------------------
# Modelimiz ImageFolder'in alfabetik sirasiyla egitildi:
#   index 0..7 -> ACIK_TEKERLEKLI, HATCHBACK, MICRO, PICK_UP,
#                 SEDAN, STATION_WAGON, SUV, VAN
#
# Test scripti ise proje dokumanindaki sirayla 1..8 bekliyor:
#   1=SUV  2=VAN  3=STATION_WAGON  4=MICRO
#   5=ACIK_TEKERLEKLI  6=SEDAN  7=HATCHBACK  8=PICK_UP
#
# Bu sozluk, modelin string etiketini proje numarasina cevirir.
MODEL_ETIKET_TO_PROJE_NO = {
    "SUV":             1,
    "VAN":             2,
    "STATION_WAGON":   3,
    "MICRO":           4,
    "ACIK_TEKERLEKLI": 5,
    "SEDAN":           6,
    "HATCHBACK":       7,
    "PICK_UP":         8,
}

# Desteklenen gorsel uzantilari
GORSEL_UZANTILARI = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ana():
    ayrastirici = argparse.ArgumentParser()
    ayrastirici.add_argument("--gorsel_klasor", type=str, required=True,
                             help="Test goruntulerinin bulundugu klasor")
    ayrastirici.add_argument("--model", type=str,
                             default="ciktilar/en_iyi_model.pth",
                             help="Egitilmis model dosyasi")
    ayrastirici.add_argument("--cikti", type=str, default="preds.txt",
                             help="Tahminlerin yazilacagi dosya")
    parametreler = ayrastirici.parse_args()

    gorsel_klasor = Path(parametreler.gorsel_klasor)
    if not gorsel_klasor.exists():
        raise FileNotFoundError(f"{gorsel_klasor} bulunamadi.")

    # Modeli bir kez yukle
    print(f"Model yukleniyor: {parametreler.model}")
    tahminleyici = Tahminleyici(parametreler.model)

    # Klasordeki tum gorselleri sirali olarak topla
    gorseller = sorted(
        p for p in gorsel_klasor.iterdir()
        if p.is_file() and p.suffix.lower() in GORSEL_UZANTILARI
    )

    if not gorseller:
        print(f"[UYARI] {gorsel_klasor} icinde gorsel bulunamadi.")
        return

    print(f"{len(gorseller)} gorsel bulundu. Tahmin yapiliyor...\n")

    # Her gorsel icin tahmin yap ve satirlari biriktir
    satirlar = []
    for gorsel_yolu in gorseller:
        try:
            etiket, guven, _ = tahminleyici.tahmin_et(gorsel_yolu)
            proje_no = MODEL_ETIKET_TO_PROJE_NO[etiket]
            # Test scriptinin bekledigi format: "dosya | label: SAYI"
            satir = f"{gorsel_yolu.name} | label: {proje_no}"
            satirlar.append(satir)
            print(f"  {gorsel_yolu.name:40s} -> {etiket:16s} "
                  f"(no={proje_no}, guven={guven:.2%})")
        except Exception as hata:
            print(f"  [HATA] {gorsel_yolu.name}: {hata}")
            continue

    # preds.txt dosyasina yaz
    cikti_yolu = Path(parametreler.cikti)
    with open(cikti_yolu, "w", encoding="utf-8") as f:
        f.write("\n".join(satirlar) + "\n")

    print(f"\nTamamlandi. {len(satirlar)} tahmin '{cikti_yolu}' "
          f"dosyasina yazildi.")


if __name__ == "__main__":
    ana()
