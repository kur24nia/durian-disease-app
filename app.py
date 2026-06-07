import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = 'durian_efficientnetb1_finetune.keras'
IMG_SIZE   = (240, 240)

CLASS_NAMES = {
    0: 'Leaf_Algal',
    1: 'Leaf_Blight',
    2: 'Leaf_Colletotrichum',
    3: 'Leaf_Healthy',
    4: 'Leaf_Phomopsis',
    5: 'Leaf_Rhizoctonia'
}

CLASS_INFO = {
    'Leaf_Algal': {
        'emoji': '🟢',
        'desc': 'Penyakit alga pada daun durian. Ditandai dengan bercak hijau atau oranye pada permukaan daun.',
        'color': '#2ecc71'
    },
    'Leaf_Blight': {
        'emoji': '🟤',
        'desc': 'Penyakit hawar daun. Ditandai dengan bercak coklat yang meluas dan mengering pada tepi daun.',
        'color': '#e67e22'
    },
    'Leaf_Colletotrichum': {
        'emoji': '🔴',
        'desc': 'Penyakit antraknosa oleh jamur Colletotrichum. Ditandai dengan bercak coklat gelap dengan tepi kuning.',
        'color': '#e74c3c'
    },
    'Leaf_Healthy': {
        'emoji': '✅',
        'desc': 'Daun durian dalam kondisi sehat. Tidak ditemukan tanda-tanda penyakit.',
        'color': '#27ae60'
    },
    'Leaf_Phomopsis': {
        'emoji': '🟠',
        'desc': 'Penyakit Phomopsis pada daun durian. Ditandai dengan bercak coklat dengan pusat abu-abu.',
        'color': '#f39c12'
    },
    'Leaf_Rhizoctonia': {
        'emoji': '🟣',
        'desc': 'Penyakit Rhizoctonia pada daun durian. Ditandai dengan bercak coklat kemerahan.',
        'color': '#9b59b6'
    }
}

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict(image, model):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)

    # Pastikan RGB
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # EfficientNetB1 tidak perlu rescale
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    start = time.time()
    predictions = model.predict(img_array, verbose=0)
    elapsed = (time.time() - start) * 1000

    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100

    return CLASS_NAMES[predicted_class], confidence, predictions[0], elapsed

# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(
    page_title='Klasifikasi Penyakit Daun Durian',
    page_icon='🌿',
    layout='wide'
)

# Header
st.title('🌿 Klasifikasi Penyakit Daun Durian')
st.markdown('**Model:** EfficientNetB1 | **Akurasi:** 90.61% | **Dataset:** Durian Leaf Disease Vietnam 2025')
st.divider()

# Load model
with st.spinner('Memuat model...'):
    try:
        model = load_model()
        st.success('Model berhasil dimuat!')
    except Exception as e:
        st.error(f'Gagal memuat model: {e}')
        st.stop()

# Layout 2 kolom
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader('📤 Upload Gambar Daun Durian')
    uploaded_file = st.file_uploader(
        'Pilih gambar daun durian',
        type=['jpg', 'jpeg', 'png'],
        help='Upload gambar daun durian dalam format JPG atau PNG'
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diupload', use_container_width=True)

with col2:
    st.subheader('📊 Hasil Klasifikasi')

    if uploaded_file:
        with st.spinner('Menganalisis gambar...'):
            pred_class, confidence, all_probs, elapsed = predict(image, model)

        info = CLASS_INFO[pred_class]

        # Hasil utama
        st.markdown(f"""
        <div style='background-color: {info["color"]}22; padding: 20px; border-radius: 10px; border-left: 5px solid {info["color"]}'>
            <h2 style='color: {info["color"]}; margin:0'>{info["emoji"]} {pred_class}</h2>
            <h3 style='margin:5px 0'>Confidence: {confidence:.2f}%</h3>
            <p style='margin:5px 0'>{info["desc"]}</p>
            <small>⏱️ Waktu inferensi: {elapsed:.2f} ms</small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('---')

        # Probabilitas semua kelas
        st.markdown('**Probabilitas per Kelas:**')
        for i, (cls_name, prob) in enumerate(zip(CLASS_NAMES.values(), all_probs)):
            col_name, col_bar = st.columns([1, 2])
            with col_name:
                st.write(cls_name)
            with col_bar:
                st.progress(float(prob))
                st.caption(f'{prob*100:.2f}%')
    else:
        st.info('👆 Upload gambar daun durian untuk memulai klasifikasi')

# Info kelas
st.divider()
st.subheader('📚 Informasi Kelas Penyakit')
cols = st.columns(3)
for i, (cls_name, info) in enumerate(CLASS_INFO.items()):
    with cols[i % 3]:
        st.markdown(f"""
        <div style='background-color: {info["color"]}11; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid {info["color"]}44'>
            <b>{info["emoji"]} {cls_name}</b><br>
            <small>{info["desc"]}</small>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption('Dibuat untuk keperluan penelitian klasifikasi penyakit daun durian menggunakan EfficientNetB1')
