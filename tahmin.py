"""
tahmin.py
=========
Egitilmis modeli yukleyip tek goruntuden tahmin donduren modul.

Hem web arayuzu (arayuz.py) hem de hocalarin paylasacagi test scripti
bu modulu ice aktararak kullanabilir:

    from tahmin import Tahminleyici
    t = Tahminleyici("ciktilar/en_iyi_model.pth")
    etiket, guven, olasiliklar = t.tahmin_et("araba.jpg")

Komut satirindan da test edilebilir:
    python tahmin.py --model ciktilar/en_iyi_model.pth --gorsel araba.jpg
"""

import argparse
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


class Tahminleyici:
    """
    Tek bir model kontrol noktasi (checkpoint) uzerinden tahmin yapan sinif.

    Checkpoint icerigi:
        - state_dict       : model agirliklari
        - sinif_isimleri   : sinif isimleri (sirali)
        - gorsel_boyutu    : egitimde kullanilan giris boyutu
        - mimari           : mimari adi (kontrol icin)
    """

    def __init__(self, model_yolu: Union[str, Path],
                 cihaz: str = None):
        # Cihaz secimi: GPU varsa kullan, yoksa CPU
        if cihaz is None:
            cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        self.cihaz = torch.device(cihaz)

        # weights_only=False cunku sinif_isimleri gibi meta veri de saklanmis
        kontrol_noktasi = torch.load(model_yolu,
                                     map_location=self.cihaz,
                                     weights_only=False)
        self.sinif_isimleri = kontrol_noktasi["sinif_isimleri"]
        self.gorsel_boyutu = kontrol_noktasi.get("gorsel_boyutu", 224)
        self.sinif_sayisi = len(self.sinif_isimleri)

        # Modeli olustur (rastgele agirliklarla) ve sonra checkpoint'ten yukle
        # weights=None cunku kendi agirliklarimi yukleyecegim.
        self.model = efficientnet_b0(weights=None)
        giris_ozellik_sayisi = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(giris_ozellik_sayisi, self.sinif_sayisi),
        )
        self.model.load_state_dict(kontrol_noktasi["state_dict"])
        self.model.to(self.cihaz)
        self.model.eval()

        # Egitimdeki "degerlendirme_donusumu" ile birebir ayni - tutarlilik sart!
        # Aksi halde egitim ve tahmin arasinda dagilim farki olur.
        self.donusum = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.gorsel_boyutu),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def tahmin_et(self, gorsel: Union[str, Path, Image.Image]
                  ) -> Tuple[str, float, dict]:
        """
        Bir goruntu icin tahmin dondurur.

        Args:
            gorsel: dosya yolu, Path veya acik PIL.Image
        Returns:
            etiket          : en yuksek olasilikli sinif adi (str)
            guven           : o sinifin olasiligi (0-1 arasi float)
            olasiliklar     : {sinif_adi: olasilik} sozlugu (8 anahtar)
        """
        # Girdiyi PIL Image'e cevir, RGB'ye normalize et (PNG alpha vs.)
        if isinstance(gorsel, (str, Path)):
            resim = Image.open(gorsel).convert("RGB")
        elif isinstance(gorsel, Image.Image):
            resim = gorsel.convert("RGB")
        else:
            raise TypeError(f"Desteklenmeyen tur: {type(gorsel)}")

        # Donusum + batch boyutu ekle (1, 3, 224, 224)
        x = self.donusum(resim).unsqueeze(0).to(self.cihaz)
        logitler = self.model(x)
        # softmax ile log-olasiliklari olasiliga cevir
        olasilik_tensoru = torch.softmax(logitler, dim=1)[0]
        olasiliklar_dizi = olasilik_tensoru.cpu().numpy()

        # En yuksek olasilikli sinifi bul
        en_yuksek_indeks = int(olasiliklar_dizi.argmax())
        etiket = self.sinif_isimleri[en_yuksek_indeks]
        guven = float(olasiliklar_dizi[en_yuksek_indeks])

        olasiliklar = {ad: float(olasiliklar_dizi[i])
                       for i, ad in enumerate(self.sinif_isimleri)}
        return etiket, guven, olasiliklar


def ana():
    """Komut satiri kullanimi - tek bir gorsel uzerinde hizli test."""
    ayrastirici = argparse.ArgumentParser()
    ayrastirici.add_argument("--model", type=str,
                             default="ciktilar/en_iyi_model.pth")
    ayrastirici.add_argument("--gorsel", type=str, required=True)
    parametreler = ayrastirici.parse_args()

    tahminleyici = Tahminleyici(parametreler.model)
    etiket, guven, olasiliklar = tahminleyici.tahmin_et(parametreler.gorsel)

    print(f"\nTahmin: {etiket}  (guven: {guven:.2%})\n")
    print("Tum sinif olasiliklari:")
    for ad, olasilik in sorted(olasiliklar.items(), key=lambda kv: -kv[1]):
        cubuk = "█" * int(olasilik * 30)
        print(f"  {ad:18s} {olasilik:6.2%}  {cubuk}")


if __name__ == "__main__":
    ana()
