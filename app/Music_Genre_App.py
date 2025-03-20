import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------------
# Cache Model Loading
# -----------------------------------
@st.cache_resource
def load_model():
    #model = tf.keras.models.load_model('./Trained_model.keras')
    model = tf.keras.models.load_model('models/Trained_model.keras')
    return model

# ------------------------------------
# Load and Preprocess Audio File
# ------------------------------------
def load_and_preprocess_file(file_obj, chunk_duration, overlap_duration, target_shape=(150, 150)):
    data = []
    
    try:
        audio_data, sampling_rate = librosa.load(file_obj, sr=None)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return np.array([]), None

    chunk_samples = int(chunk_duration * sampling_rate)
    overlap_samples = int(overlap_duration * sampling_rate)
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    progress = st.progress(0)

    for j in range(num_chunks):
        start = j * (chunk_samples - overlap_samples)
        end = min(start + chunk_samples, len(audio_data))
        chunk = audio_data[start:end]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sampling_rate)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
        mel_spectrogram_resized = tf.image.resize(mel_spectrogram, target_shape).numpy()

        data.append(mel_spectrogram_resized)

        progress.progress((j + 1) / num_chunks)

    progress.empty()

    return np.array(data), sampling_rate

# ------------------------------------
# Model Prediction Function
# ------------------------------------
def model_prediction(model, X_test):
    y_pred = model.predict(X_test)
    
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)

    max_count = np.max(counts)
    max_element = unique_elements[counts == max_count]

    confidence = np.max(tf.reduce_max(y_pred, axis=1))

    return int(max_element[0]), float(confidence)

# ------------------------------------
# Genre Descriptions
# ------------------------------------
genre_descriptions = {
    'blues': "🎸 Blues is known for its soulful melodies, emotional depth, and expressive guitar work.",
    'classical': "🎻 Classical music features orchestral arrangements, complex structures, and timeless compositions.",
    'country': "🤠 Country blends folk tunes with storytelling lyrics and twangy guitars.",
    'disco': "💃 Disco brings upbeat rhythms, groovy bass lines, and dancefloor energy.",
    'hiphop': "🎤 Hip-Hop combines rhythmic beats, rap vocals, and strong lyrical content.",
    'jazz': "🎷 Jazz thrives on improvisation, swing rhythms, and complex chord progressions.",
    'metal': "🤘 Metal features heavy guitar riffs, powerful drumming, and intense vocals.",
    'pop': "🎤 Pop music offers catchy melodies, repetitive hooks, and broad appeal.",
    'reggae': "🌴 Reggae is characterized by its offbeat rhythms, relaxed vibe, and socially conscious lyrics.",
    'rock': "🎸 Rock combines electric guitar-driven sound with energetic vocals and powerful beats."
}

def show_genre_description(predicted_genre):
    description = genre_descriptions[predicted_genre]
    st.markdown(f"""
        <div style="
            background-color: #ff4b4b;
            padding: 15px;
            border-radius: 10px;
            color: white;
            font-size: 18px;
            text-align: center;">
            <strong>🎵 About {predicted_genre.capitalize()} 🎵</strong><br><br>
            {description}
        </div>
    """, unsafe_allow_html=True)

# ------------------------------------
# Sidebar Navigation (Design Inspired)
# ------------------------------------
st.sidebar.title("🎛️ Dashboard")

app_mode = st.sidebar.selectbox("Select Page", ["🏠 Home", "ℹ️ About Project", "🎼 Prediction"])

if app_mode == "🎼 Prediction":
    st.sidebar.markdown("---")
    chunk_duration = st.sidebar.slider("Chunk Duration (seconds)", 1, 10, 4)
    overlap_duration = st.sidebar.slider("Overlap Duration (seconds)", 0, 5, 2)
else:
    chunk_duration = 4
    overlap_duration = 2

# ------------------------------------
# Home Page (Design Inspired)
# ------------------------------------
if app_mode == "🏠 Home":
    st.markdown("""
        <style>
        .stApp {
            background-color: #181646;
            color: white;
        }
        h2, h3 {
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🎶 Welcome to the Music Genre Classification System! 🎧")
    #st.image("music_genre_home.png", use_container_width=True)
    image = Image.open('app/assets/image.png')
    st.image(image)

    st.markdown("""
        **Our goal is to help identify music genres from audio tracks efficiently. Upload an audio file, and our AI-powered system will analyze it to detect its genre. Discover the power of AI in music analysis!**

        ### How It Works
        1. **Upload Audio**: Go to the **Prediction** page and upload an audio file.
        2. **Analysis**: Our AI will process the audio and classify it into one of the 10 predefined genres.
        3. **Results**: View the predicted genre along with details and visual feedback.

        ### Why Choose Us?
        - 🎯 **Accuracy**: State-of-the-art deep learning models.
        - 🖱️ **User-Friendly**: Simple and intuitive interface.
        - ⚡ **Fast & Efficient**: Quick results for rapid exploration.

        ### Ready to Explore?
        👉 Head to the **Prediction** page via the sidebar and upload your track!
    """)

# ------------------------------------
# About Project Page (Design Inspired)
# ------------------------------------
elif app_mode == "ℹ️ About Project":
    st.markdown("""
        ### About the Project 🎶
        Understanding sound and differentiating one song from another has fascinated experts for decades. This project explores how AI can classify music based on acoustic and spectral features.

        ### Dataset 📁
        - **GTZAN Music Genre Collection**: 10 genres, 100 tracks per genre, each 30 seconds long.
        - **Genres**: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock.
        - **Spectrograms**: Audio files converted to Mel Spectrogram images to feed the CNN model.

        ### Technical Details 🛠️
        - **Audio Preprocessing**: Audio segmented into overlapping 4-second chunks.
        - **Model**: Convolutional Neural Network (CNN) trained on spectrograms.
        - **Prediction**: Aggregated predictions for more accurate results.

        ### Future Scope 🚀
        - More modern genres (EDM, K-Pop, Lo-Fi)
        - Multi-label classification support
        - Real-time streaming predictions

        #### Join us in bridging the gap between music and technology!
    """)

# ------------------------------------
# Prediction Page (Design + Functionality)
# ------------------------------------
elif app_mode == "🎼 Prediction":
    st.markdown("""
        <style>
        .stApp {
            background-color: #181646;
            color: white;
        }
        h2, h3 {
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("🎵 Genre Classification Prediction")

    # Load Model
    model = load_model()

    # Upload Audio File
    test_mp3 = st.file_uploader("Upload an audio file (.mp3 or .wav)", type=["mp3", "wav"])

    if test_mp3 is not None:
        if st.button("▶️ Play Audio"):
            st.audio(test_mp3)

        if st.button("Predict Genre 🎯"):
            with st.spinner("Analyzing your track... Please wait!"):
                X_test, sampling_rate = load_and_preprocess_file(
                    test_mp3,
                    chunk_duration=chunk_duration,
                    overlap_duration=overlap_duration
                )

                if X_test.size == 0:
                    st.error("No valid data found in the audio file.")
                else:
                    result_index, confidence = model_prediction(model, X_test)

                    #st.balloons()

                    label = [
                        'blues', 'classical', 'country', 'disco', 'hiphop', 
                        'jazz', 'metal', 'pop', 'reggae', 'rock'
                    ]

                    predicted_genre = label[result_index]

                    st.markdown(f"🎶 **Predicted Genre:** :red[{predicted_genre.capitalize()}]")
                    st.info(f"✅ **Confidence Score:** :green[{confidence:.2f}]")

                    show_genre_description(predicted_genre)

                    # Show Mel Spectrogram Visualization
                    st.subheader("🎛️ Mel Spectrogram Visualization (First Chunk)")
                    mel_spectrogram = X_test[0].squeeze()

                    fig, ax = plt.subplots()
                    img = librosa.display.specshow(
                        mel_spectrogram,
                        sr=sampling_rate if sampling_rate else 22050,
                        x_axis='time',
                        y_axis='mel',
                        ax=ax
                    )
                    ax.set_title('Mel Spectrogram (First Chunk)')
                    fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    st.pyplot(fig)
