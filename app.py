import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="DoAR | Dompet Anak Rantau",
    page_icon="💸",
    layout="wide",
)

# =========================================================
# CUSTOM CSS / THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    .hero {
        background: linear-gradient(135deg, #0F9D8B 0%, #0B6E63 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(15,157,139,0.25);
    }
    .hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.2rem;
        color: white;
    }
    .hero p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }

    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 16px rgba(17,24,39,0.06);
        border-left: 5px solid #0F9D8B;
        margin-bottom: 0.6rem;
    }
    .metric-card.accent {
        border-left: 5px solid #FF8A3D;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #6B7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        font-family: 'Poppins', sans-serif;
    }

    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
        box-shadow: 0 4px 16px rgba(17,24,39,0.05);
        margin-bottom: 1.2rem;
    }

    .status-card {
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        margin-bottom: 1rem;
    }
    .status-aman { background:#DCFCE7; color:#166534; border:1px solid #86EFAC; }
    .status-defisit { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
    .status-card .big {
        font-size: 1.8rem;
        font-weight: 800;
    }

    .tip-box {
        background: #F0FDFA;
        border-left: 4px solid #0F9D8B;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.94rem;
        line-height: 1.6;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #0F9D8B, #0B6E63);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 4px 14px rgba(15,157,139,0.35);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #12B39E, #0F9D8B);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# =========================================================
# HERO HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>💸 DoAR — Dompet Anak Rantau</h1>
        <p>Kelola uang bulanan dan pantau risiko keuangan kamu dengan prediksi berbasis Naive Bayes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
# =========================================================
# FUNGSI BANTU
# =========================================================
def kategorikan_rasio(rasio, batas_rendah, batas_sedang):
    if rasio < batas_rendah:
        return "Rendah"
    elif rasio < batas_sedang:
        return "Sedang"
    return "Tinggi"

def kategorikan_nominal(nominal, batas_rendah, batas_tinggi):
    if nominal < batas_rendah:
        return "Rendah"
    elif nominal <= batas_tinggi:
        return "Sedang"
    return "Tinggi"

def metric_card(label, value, accent=False):
    cls = "metric-card accent" if accent else "metric-card"
    st.markdown(
        f"""<div class="{cls}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )

# MEMBACA DATASET ASLI
@st.cache_data
def muat_dataset_asli():
    return pd.read_csv("dataset_keuangan.csv")

@st.cache_resource
def latih_model(df):
    X = df[["Uang_Saku", "Saving", "Kebutuhan_Utama", "Lifestyle", "Makan_Luar"]]
    y = df["Status_Keuangan"]
    encoder = OrdinalEncoder(categories=[["Rendah", "Sedang", "Tinggi"]] * 5)
    X_encoded = encoder.fit_transform(X)
    model = CategoricalNB()
    model.fit(X_encoded, y)
    return model, encoder

PALET = ["#0F9D8B", "#FF8A3D", "#3B82F6", "#F59E0B", "#8B5CF6", "#EC4899"]

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["🏦  Alokasi Anggaran", "🔍  Prediksi Risiko"])

# ---------------------------------------------------------
# TAB 1 — ALOKASI ANGGARAN
# ---------------------------------------------------------
with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col_input, col_metric = st.columns([1.1, 1])

    with col_input:
        uang_saku = st.number_input(
            "Total Uang Saku Bulanan (Rp)", min_value=0, value=2_000_000, step=100_000
        )
        persen_saving = st.slider("Target Uang Saving / Dana Darurat (%)", 0, 50, 20)

    nominal_saving = uang_saku * (persen_saving / 100)
    batas_kebutuhan = uang_saku - nominal_saving

    with col_metric:
        metric_card("Uang Saving (diolah terpisah)", f"Rp {nominal_saving:,.0f}")
        metric_card("Batas Maks. Kebutuhan Utama", f"Rp {batas_kebutuhan:,.0f}", accent=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Rencana Pengeluaran Kebutuhan Utama")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        makan = st.number_input("🍚 Makan & Minum", value=800_000, step=50_000)
    with c2:
        lifestyle = st.number_input("☕ Lifestyle", value=300_000, step=50_000)
    with c3:
        transport = st.number_input("🛵 Transportasi", value=200_000, step=50_000)
    with c4:
        internet = st.number_input("📶 Internet", value=100_000, step=50_000)
    with c5:
        lainnya = st.number_input("🎒 Keperluan Kuliah", value=200_000, step=50_000)

    kebutuhan_esensial = transport + internet + lainnya
    total_kebutuhan = makan + lifestyle + kebutuhan_esensial
    sisa = batas_kebutuhan - total_kebutuhan

    persen_terpakai = min(total_kebutuhan / batas_kebutuhan, 1.0) if batas_kebutuhan > 0 else 0
    st.write("")
    st.progress(persen_terpakai, text=f"Pemakaian anggaran: {persen_terpakai*100:.0f}%")

    if total_kebutuhan > batas_kebutuhan:
        st.warning(
            f"⚠️ Total pengeluaran (Rp {total_kebutuhan:,.0f}) melebihi batas kebutuhan "
            f"utama (Rp {batas_kebutuhan:,.0f}) sebesar **Rp {-sisa:,.0f}**!"
        )
    else:
        st.success(
            f"✅ Total pengeluaran (Rp {total_kebutuhan:,.0f}) berada dalam batas aman. "
            f"Sisa anggaran: **Rp {sisa:,.0f}**"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Visualisasi Alokasi")
    labels_all = ["Saving", "Makan", "Lifestyle", "Transport", "Internet", "Kuliah/Kost"]
    values_all = [nominal_saving, makan, lifestyle, transport, internet, lainnya]
    labels = [l for l, v in zip(labels_all, values_all) if v > 0]
    values = [v for v in values_all if v > 0]

    if values:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.55,
                    marker=dict(colors=PALET, line=dict(color="white", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=13, family="Inter"),
                )
            ]
        )
        fig.update_layout(
            showlegend=True,
            margin=dict(t=10, b=10, l=10, r=10),
            height=380,
            annotations=[
                dict(
                    text=f"Rp {uang_saku:,.0f}",
                    x=0.5, y=0.5, font_size=16, showarrow=False,
                    font=dict(family="Poppins", color="#111827"),
                )
            ],
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2 — PREDIKSI RISIKO
# ---------------------------------------------------------
with tab1:
    pass

with tab2:
    analisis = st.button("🔍 Analisis Risiko Keuangan", type="primary")

    if analisis:
        try:
            # Membaca dataset asli secara otomatis
            df = muat_dataset_asli()
            sumber = "dataset_keuangan.csv (Data Responden Asli)"

            model, encoder = latih_model(df)

            rasio_saving = persen_saving / 100
            kat_saving = kategorikan_rasio(rasio_saving, 0.10, 0.20)
            kat_uang_saku = kategorikan_nominal(uang_saku, 1_500_000, 2_500_000)
            kat_kebutuhan = kategorikan_rasio(
                kebutuhan_esensial / uang_saku if uang_saku else 0, 0.60, 0.80
            )
            kat_lifestyle = kategorikan_rasio(
                lifestyle / uang_saku if uang_saku else 0, 0.10, 0.20
            )
            kat_makan = kategorikan_rasio(
                makan / uang_saku if uang_saku else 0, 0.30, 0.50
            )

            input_baru = pd.DataFrame(
                [[kat_uang_saku, kat_saving, kat_kebutuhan, kat_lifestyle, kat_makan]],
                columns=["Uang_Saku", "Saving", "Kebutuhan_Utama", "Lifestyle", "Makan_Luar"],
            )
            input_encoded = encoder.transform(input_baru)

            prediksi = model.predict(input_encoded)[0]
            proba = model.predict_proba(input_encoded)[0]
            proba_df = pd.DataFrame(
                {"Status": model.classes_, "Probabilitas": proba}
            ).sort_values("Probabilitas", ascending=True)

            ikon = {"Aman": "🟢", "Defisit": "🔴"}
            kelas_css = {"Aman": "status-aman", "Defisit": "status-defisit"}

            st.markdown(
                f"""<div class="status-card {kelas_css.get(prediksi, '')}">
                        <div style="font-size:0.9rem;letter-spacing:0.05em;text-transform:uppercase;">Status Prediksi Keuangan</div>
                        <div class="big">{ikon.get(prediksi, '')} {prediksi}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

            colp1, colp2 = st.columns([1, 1.2])
            with colp1:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.write("**Kategori Input Keuangan Anda:**")
                st.table(input_baru.T.rename(columns={0: "Kategori"}))
                st.markdown("</div>", unsafe_allow_html=True)

            with colp2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.write("**Probabilitas Tiap Kelas (Naive Bayes):**")
                warna_bar = {"Aman": "#16A34A", "Defisit": "#DC2626"}
                fig_bar = go.Figure(
                    go.Bar(
                        x=proba_df["Probabilitas"],
                        y=proba_df["Status"],
                        orientation="h",
                        marker_color=[warna_bar.get(s, "#0F9D8B") for s in proba_df["Status"]],
                        text=[f"{p*100:.1f}%" for p in proba_df["Probabilitas"]],
                        textposition="outside",
                    )
                )
                fig_bar.update_layout(
                    height=220,
                    margin=dict(t=10, b=10, l=10, r=30),
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.write("💡 Rekomendasi untuk Kamu:")
            if prediksi == "Defisit":
                isi = (
                    " ⚠️ Peringatan Kantong Kering! Pengeluaran kamu tergolong boros.\n"
                    " Segera kurangi pengeluaran lifestyle dan frekuensi makan di luar."
                    " Tingkatkan alokasi tabungan (saving) minimal 15–20% dari uang saku."
                )
            else:
                isi = (
                    " 🎉 JOS JIS LEK PERTAHANKAN 🎉\n"
                    " anggaran kamu aman, perbanyak lagi porsi menabung!"
                )
            st.markdown(f'<div class="tip-box">{isi.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("ℹ️ Detail Model Training"):
                st.json(
                    {
                        "Sumber Data": sumber,
                        "Jumlah Data Latih Responden": len(df),
                        "Kelas Target": list(model.classes_),
                    }
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat dataset `dataset_keuangan.csv`: {e}")

st.caption("Aplikasi DoAR — Model diprediksi secara real-time menggunakan algoritma Naive Bayes berdasarkan dataset responden mahasiswa rantau.")