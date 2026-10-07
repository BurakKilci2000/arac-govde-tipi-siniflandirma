"""
arayuz.py
=========
Streamlit ile araba govde tipi siniflandirma web arayuzu.

Calistirma:
    streamlit run arayuz.py

Proje sartlari (rapordaki bolumler):
    Bolum 1: Goruntu yukleme alani + onizleme
    Bolum 2: "Tahmin Yap" butonu
    Bolum 3: Tahmin sonucu + tum sinif olasiliklari (bar chart)
    Bolum 4: Yuklenen gorsel ile sonuc yan yana
"""

import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from tahmin import Tahminleyici


# Sayfa konfigurasyonu (genis duzen, baslik, ikon)
st.set_page_config(
    page_title="Araba Govde Tipi Siniflandirici",
    page_icon="🚗",
    layout="wide",
)

# Arayuzde gorunecek Turkce sinif isimleri (kod icindekilerden ayri tutuldu)
GORUNTULENEN_ISIMLER = {
    "SUV":             "SUV",
    "VAN":             "Van",
    "STATION_WAGON":   "Station Wagon",
    "MICRO":           "Micro",
    "ACIK_TEKERLEKLI": "Acik Tekerlekli (F1)",
    "SEDAN":           "Sedan",
    "HATCHBACK":       "Hatchback",
    "PICK_UP":         "Pick-up",
}

MODEL_YOLU = "ciktilar/en_iyi_model.pth"


# Modeli her sayfa yenilemesinde tekrar yuklememek icin cache'le
@st.cache_resource(show_spinner="Model yukleniyor...")
def tahminleyiciyi_yukle(yol: str) -> Tahminleyici:
    return Tahminleyici(yol)


def ana():
    st.title("🚗 Araba Govde Tipi Siniflandirma")
    st.caption("Kocaeli Universitesi - Yazilim Laboratuvari II - Proje III")

    # Modelin var olup olmadigini kontrol et
    if not Path(MODEL_YOLU).exists():
        st.error(
            f"Model dosyasi bulunamadi: `{MODEL_YOLU}`\n\n"
            "Once `python egitim.py` ile modeli egitmeniz gerekiyor."
        )
        st.stop()

    tahminleyici = tahminleyiciyi_yukle(MODEL_YOLU)

    # ----------------------------------------------------------------
    # BOLUM 1: Goruntu yukleme alani
    # ----------------------------------------------------------------
    st.subheader("1. Gorsel Yukleyin")
    yuklenen = st.file_uploader(
        "Bir araba gorseli secin (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    if yuklenen is None:
        st.info("Tahmin yapmak icin yukaridan bir gorsel yukleyin.")
        st.stop()

    # PIL ile ac, RGB'ye cevir (PNG alpha kanali vs. olabilir)
    resim = Image.open(yuklenen).convert("RGB")

    # ----------------------------------------------------------------
    # BOLUM 2 & 4: Gorsel sol, sonuclar sag tarafta yan yana
    # ----------------------------------------------------------------
    sol_sutun, sag_sutun = st.columns([1, 1])

    with sol_sutun:
        st.markdown("**Yuklenen Gorsel**")
        st.image(resim, use_container_width=True, caption=yuklenen.name)

        # Tahmin butonu
        tahmin_tiklandi = st.button("🔍 Tahmin Yap",
                                    type="primary",
                                    use_container_width=True)

    with sag_sutun:
        if not tahmin_tiklandi:
            st.info("Sol taraftaki **Tahmin Yap** butonuna basin.")
            return

        # ------------------------------------------------------------
        # BOLUM 3: Tahmin sonucu ciktisi
        # ------------------------------------------------------------
        # Cikarim suresini olc (proje "hiz onemli" diyor)
        baslangic = time.perf_counter()
        etiket, guven, olasiliklar = tahminleyici.tahmin_et(resim)
        gecen_ms = (time.perf_counter() - baslangic) * 1000

        gosterilen_etiket = GORUNTULENEN_ISIMLER.get(etiket, etiket)

        st.markdown("**Tahmin Sonucu**")

        # Buyuk ve belirgin sonuc gosterimi
        st.markdown(
            f"""
            <div style="background-color:#1f77b4; padding:24px;
                        border-radius:12px; text-align:center; color:white;">
                <div style="font-size:14px; opacity:0.85;">
                    Tahmin Edilen Sinif
                </div>
                <div style="font-size:42px; font-weight:700; margin-top:8px;">
                    {gosterilen_etiket}
                </div>
                <div style="font-size:18px; margin-top:8px;">
                    Guven: <b>{guven:.2%}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hiz metrigi
        st.caption(f"⏱️ Cikarim suresi: {gecen_ms:.1f} ms")

        # Tum siniflar icin olasilik dagilimi (bar chart)
        st.markdown("**Tum Siniflar Icin Olasilik Dagilimi**")
        df = pd.DataFrame({
            "Sinif": [GORUNTULENEN_ISIMLER.get(k, k) for k in olasiliklar.keys()],
            "Olasilik": list(olasiliklar.values()),
        }).sort_values("Olasilik", ascending=False)

        # Her sinif icin bir progress bar - okunabilir bar chart sunumu
        for _, satir in df.iterrows():
            st.write(f"{satir['Sinif']}  —  {satir['Olasilik']:.2%}")
            st.progress(float(satir["Olasilik"]))

    # ----------------------------------------------------------------
    # Sayfa sonu: yardimci bilgi paneli (sol kenar cubugu)
    # ----------------------------------------------------------------
    with st.sidebar:
        st.header("Model Bilgisi")
        st.write(f"**Mimari:** EfficientNet-B0")
        st.write(f"**Sinif sayisi:** {len(tahminleyici.sinif_isimleri)}")
        st.write(f"**Giris boyutu:** "
                 f"{tahminleyici.gorsel_boyutu}×{tahminleyici.gorsel_boyutu}")
        st.write(f"**Cihaz:** `{tahminleyici.cihaz}`")
        st.divider()
        st.caption("Hazirlayan: Burak Kılcı")


if __name__ == "__main__":
    ana()
