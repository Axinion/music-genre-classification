import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from tensorflow.image import resize
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from pydub import AudioSegment
import io
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import pandas as pd
from datetime import datetime
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import concurrent.futures
import time
from fpdf import FPDF
import base64
from pathlib import Path
import os

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background-color: #f5f5f5;
        color: #2c3e50;
        padding: 0.5rem;
    }
    
    /* Header Styles */
    .header-container {
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(45deg, #4CAF50, #2196F3);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        margin: 0;
        font-size: clamp(1.5rem, 5vw, 2.5rem);
        font-weight: 700;
        line-height: 1.2;
    }
    
    .header-subtitle {
        color: white;
        margin: 0.5rem 0 0 0;
        font-size: clamp(0.9rem, 3vw, 1.2rem);
        opacity: 0.9;
        line-height: 1.4;
    }
    
    /* Button Styles */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: clamp(0.9rem, 2.5vw, 1rem);
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Card Styles */
    .metric-card {
        background-color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        color: #666;
        font-size: clamp(0.8rem, 2.5vw, 1rem);
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        color: #2c3e50;
        font-size: clamp(1.2rem, 4vw, 1.8rem);
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        min-height: 2.5rem;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        gap: 0.5rem;
        padding: 0.5rem;
        font-size: clamp(0.8rem, 2.5vw, 1rem);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    
    /* File Uploader Styles */
    .stFileUploader {
        background-color: white;
        padding: 0.8rem;
        border-radius: 8px;
        border: 2px dashed #4CAF50;
    }
    
    /* Footer Styles */
    .footer {
        text-align: center;
        padding: 1.5rem 0.5rem;
        margin-top: 2rem;
        color: #666;
        font-size: clamp(0.8rem, 2.5vw, 0.9rem);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main {
            padding: 0.3rem;
        }
        
        .header-container {
            padding: 1rem 0.8rem;
            margin-bottom: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.4rem;
        }
        
        /* Adjust column layout for mobile */
        [data-testid="column"] {
            width: 100% !important;
            padding: 0.3rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Main UI
st.markdown("""
    <div class='header-container'>
        <h1 class='header-title'>🎵 Music Genre Classification System</h1>
        <p class='header-subtitle'>Upload your music and discover its genre, characteristics, and more!</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Single File Analysis", "Batch Processing", "Music Visualization"])

# Spotify Integration
def init_spotify():
    if 'spotify' not in st.session_state:
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id="YOUR_SPOTIFY_CLIENT_ID",
                client_secret="YOUR_SPOTIFY_CLIENT_SECRET"
            )
            st.session_state.spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        except:
            st.session_state.spotify = None

# Function
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./models/Trained_model.keras')
    return model

# Load the model at startup
model = load_model()

# Load and Preprocess Audio File
def load_and_preprocess_file(file_obj, target_shape=(150, 150)):
    data = []
    
    # Save the uploaded file to a temporary file
    temp_file = io.BytesIO(file_obj.getvalue())
    
    # Use librosa with audioread backend for MP3 files
    audio_data, sampling_rate = librosa.load(temp_file, sr=None, res_type='kaiser_fast')
    
    # define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = int(chunk_duration * sampling_rate)
    overlap_samples = int(overlap_duration * sampling_rate)

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for j in range(num_chunks):
        start = j * (chunk_samples - overlap_samples)
        end = min(start + chunk_samples, len(audio_data))
        chunk = audio_data[start:end]

        # Pad chunk if it's smaller than chunk_samples
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sampling_rate)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
        mel_spectrogram_resized = resize(mel_spectrogram, target_shape)
        data.append(mel_spectrogram_resized)

    return np.array(data)


# Model Prediction
def model_prediction(model, X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_element = unique_elements[counts == max_count]
    return max_element[0]


def analyze_audio_quality(audio_data, sampling_rate):
    quality_metrics = {}
    
    # 1. Calculate RMS (Root Mean Square) for overall volume
    rms = librosa.feature.rms(y=audio_data)[0]
    quality_metrics['average_volume'] = float(np.mean(rms))
    
    # 2. Detect clipping
    clipping_threshold = 0.99
    clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
    clipping_percentage = (clipped_samples / len(audio_data)) * 100
    quality_metrics['clipping_percentage'] = float(clipping_percentage)
    
    # 3. Calculate Signal-to-Noise Ratio (SNR)
    noise_floor = np.percentile(np.abs(audio_data), 10)
    signal_power = np.mean(np.square(audio_data))
    noise_power = np.mean(np.square(noise_floor))
    snr = 10 * np.log10(signal_power / noise_power)
    quality_metrics['snr_db'] = float(snr)
    
    # 4. Dynamic Range
    dynamic_range = 20 * np.log10(np.max(np.abs(audio_data)) / (np.min(np.abs(audio_data)) + 1e-10))
    quality_metrics['dynamic_range_db'] = float(dynamic_range)
    
    # 5. Frequency Spectrum Analysis
    spectrum = np.abs(librosa.stft(audio_data))
    freqs = librosa.fft_frequencies(sr=sampling_rate)
    low_freq_mask = freqs < 200
    mid_freq_mask = (freqs >= 200) & (freqs < 2000)
    high_freq_mask = freqs >= 2000
    
    low_power = float(np.mean(spectrum[low_freq_mask]))
    mid_power = float(np.mean(spectrum[mid_freq_mask]))
    high_power = float(np.mean(spectrum[high_freq_mask]))
    
    quality_metrics['frequency_balance'] = {
        'low_freq_power': low_power,
        'mid_freq_power': mid_power,
        'high_freq_power': high_power
    }
    
    return quality_metrics

def plot_audio_quality(quality_metrics):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Volume and Clipping
    ax1.bar(['Average Volume', 'Clipping %'], 
            [quality_metrics['average_volume'], quality_metrics['clipping_percentage']])
    ax1.set_title('Volume and Clipping Analysis')
    
    # 2. SNR and Dynamic Range
    ax2.bar(['SNR (dB)', 'Dynamic Range (dB)'], 
            [quality_metrics['snr_db'], quality_metrics['dynamic_range_db']])
    ax2.set_title('Signal Quality Metrics')
    
    # 3. Frequency Balance
    freq_balance = quality_metrics['frequency_balance']
    ax3.bar(['Low', 'Mid', 'High'], 
            [freq_balance['low_freq_power'], 
             freq_balance['mid_freq_power'], 
             freq_balance['high_freq_power']])
    ax3.set_title('Frequency Balance')
    
    # 4. Quality Score
    quality_score = (
        (100 - quality_metrics['clipping_percentage']) * 0.3 +
        (min(quality_metrics['snr_db'], 60) / 60) * 100 * 0.3 +
        (min(quality_metrics['dynamic_range_db'], 60) / 60) * 100 * 0.4
    )
    ax4.bar(['Overall Quality Score'], [quality_score])
    ax4.set_ylim(0, 100)
    ax4.set_title('Overall Quality Score')
    
    plt.tight_layout()
    return fig

# Function to get genre-based recommendations
def get_genre_recommendations(predicted_genre, num_recommendations=5):
    genre_similarity = {
        'blues': ['jazz', 'rock', 'soul'],
        'classical': ['orchestral', 'piano', 'opera'],
        'country': ['folk', 'bluegrass', 'americana'],
        'disco': ['funk', 'pop', 'dance'],
        'hiphop': ['rap', 'r&b', 'trap'],
        'jazz': ['blues', 'swing', 'bebop'],
        'metal': ['rock', 'hard rock', 'heavy metal'],
        'pop': ['dance', 'r&b', 'rock'],
        'reggae': ['ska', 'dub', 'dancehall'],
        'rock': ['alternative', 'indie', 'metal']
    }
    
    recommendations = []
    if predicted_genre in genre_similarity:
        recommendations = random.sample(genre_similarity[predicted_genre], 
                                     min(num_recommendations, len(genre_similarity[predicted_genre])))
    return recommendations

# Function to generate a playlist
def generate_playlist(seed_genre, num_songs=10):
    playlist = []
    current_genre = seed_genre
    
    for _ in range(num_songs):
        recommendations = get_genre_recommendations(current_genre, num_recommendations=3)
        if recommendations:
            current_genre = random.choice(recommendations)
            playlist.append(current_genre)
    
    return playlist

# Function for cross-genre analysis
def analyze_cross_genre(audio_data, sampling_rate):
    features = {}
    
    # Tempo analysis
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sampling_rate)
    features['tempo'] = float(tempo)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate)[0]
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    
    # Rhythm features
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sampling_rate)
    features['rhythm_complexity'] = float(np.std(onset_env))
    
    # Harmonic features
    chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sampling_rate)
    features['harmonic_complexity'] = float(np.mean(np.std(chroma, axis=1)))
    
    return features

# Function to predict year/era
def predict_year(audio_data, sampling_rate):
    features = analyze_cross_genre(audio_data, sampling_rate)
    
    if features['spectral_centroid_mean'] > 3000:
        era = "Modern (2010s-Present)"
    elif features['spectral_centroid_mean'] > 2500:
        era = "2000s"
    elif features['spectral_centroid_mean'] > 2000:
        era = "1990s"
    else:
        era = "Classic (Pre-1990s)"
    
    return era

# Function to detect language
def detect_language(audio_data, sampling_rate):
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
        
        if np.mean(np.abs(y_harmonic)) > 0.1:
            return "Vocal detected (Language detection requires speech recognition)"
        else:
            return "Instrumental"
    except:
        return "Unable to detect vocals"

# Function to create visualizations
def create_visualizations(audio_data, sampling_rate):
    # Downsample audio data to reduce size
    target_sr = 22050  # Standard sampling rate
    if sampling_rate != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=target_sr)
        sampling_rate = target_sr
    
    # Limit the number of points for waveform
    max_points = 10000
    if len(audio_data) > max_points:
        step = len(audio_data) // max_points
        audio_data_plot = audio_data[::step]
    else:
        audio_data_plot = audio_data

    # Create subplots with specific types for 3D plot
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "scene"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=('Waveform', 'Spectrogram', '3D Features', 'Mood Analysis', 'Beat Pattern', 'Rhythm Pattern')
    )
    
    # Waveform (downsampled)
    fig.add_trace(
        go.Scatter(y=audio_data_plot, name='Waveform'),
        row=1, col=1
    )
    
    # Spectrogram (reduced resolution)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data, n_fft=2048, hop_length=512)), ref=np.max)
    # Reduce spectrogram resolution
    D = D[:, ::4]  # Take every 4th column
    fig.add_trace(
        go.Heatmap(z=D, colorscale='Viridis', name='Spectrogram'),
        row=1, col=2
    )
    
    # 3D Features (MFCC with reduced dimensions)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=13)
    # Reduce MFCC resolution
    mfccs = mfccs[:, ::4]  # Take every 4th column
    x, y = np.meshgrid(np.arange(mfccs.shape[1]), np.arange(mfccs.shape[0]))
    fig.add_trace(
        go.Surface(x=x, y=y, z=mfccs, colorscale='Viridis', name='3D Features'),
        row=2, col=1
    )
    
    # Mood Analysis (Spectral Centroid with reduced points)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate)[0]
    if len(spectral_centroids) > max_points:
        step = len(spectral_centroids) // max_points
        spectral_centroids = spectral_centroids[::step]
    fig.add_trace(
        go.Scatter(y=spectral_centroids, name='Mood', line=dict(color='purple')),
        row=2, col=2
    )
    
    # Beat Pattern (reduced points)
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sampling_rate)
    if len(onset_env) > max_points:
        step = len(onset_env) // max_points
        onset_env = onset_env[::step]
    fig.add_trace(
        go.Scatter(y=onset_env, name='Beat Pattern', line=dict(color='green')),
        row=3, col=1
    )
    
    # Rhythm Pattern (reduced points)
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sampling_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sampling_rate)
    if len(beat_times) > max_points:
        step = len(beat_times) // max_points
        beat_times = beat_times[::step]
    fig.add_trace(
        go.Scatter(x=beat_times, y=np.ones_like(beat_times), mode='markers', 
                  name='Rhythm Pattern', marker=dict(size=8, color='red')),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=False,
        scene=dict(
            xaxis_title='Time',
            yaxis_title='MFCC Coefficients',
            zaxis_title='Magnitude'
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Time (samples)', row=1, col=1)
    fig.update_xaxes(title_text='Time (frames)', row=1, col=2)
    fig.update_xaxes(title_text='Time (s)', row=2, col=2)
    fig.update_xaxes(title_text='Time (frames)', row=3, col=1)
    fig.update_xaxes(title_text='Time (s)', row=3, col=2)
    
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Frequency (Hz)', row=1, col=2)
    fig.update_yaxes(title_text='Spectral Centroid', row=2, col=2)
    fig.update_yaxes(title_text='Onset Strength', row=3, col=1)
    fig.update_yaxes(title_text='Beat Markers', row=3, col=2)
    
    return fig

# Function to generate PDF report
def generate_pdf_report(audio_data, sampling_rate, predicted_genre, quality_metrics, cross_genre_features):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Music Analysis Report', 0, 1, 'C')
    
    # Genre
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f'Predicted Genre: {predicted_genre}', 0, 1)
    
    # Quality Metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Quality Metrics:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Signal-to-Noise Ratio: {quality_metrics["snr_db"]:.1f} dB', 0, 1)
    pdf.cell(0, 10, f'Dynamic Range: {quality_metrics["dynamic_range_db"]:.1f} dB', 0, 1)
    pdf.cell(0, 10, f'Clipping Percentage: {quality_metrics["clipping_percentage"]:.1f}%', 0, 1)
    
    # Musical Characteristics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Musical Characteristics:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Tempo: {cross_genre_features["tempo"]:.1f} BPM', 0, 1)
    pdf.cell(0, 10, f'Harmonic Complexity: {cross_genre_features["harmonic_complexity"]:.2f}', 0, 1)
    pdf.cell(0, 10, f'Rhythm Complexity: {cross_genre_features["rhythm_complexity"]:.2f}', 0, 1)
    
    return pdf.output(dest='S').encode('latin1')

# Function to process a single file
def process_single_file(file_obj):
    audio_data, sampling_rate = librosa.load(file_obj, sr=None)
    X_test = load_and_preprocess_file(file_obj)
    result_index = model_prediction(model, X_test)
    label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    predicted_genre = label[result_index]
    
    quality_metrics = analyze_audio_quality(audio_data, sampling_rate)
    cross_genre_features = analyze_cross_genre(audio_data, sampling_rate)
    
    return {
        'genre': predicted_genre,
        'quality_metrics': quality_metrics,
        'cross_genre_features': cross_genre_features,
        'audio_data': audio_data,
        'sampling_rate': sampling_rate
    }

# Function to process multiple files
def process_multiple_files(file_objs):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file_obj): file_obj for file_obj in file_objs}
        for future in concurrent.futures.as_completed(future_to_file):
            results.append(future.result())
    return results

with tab1:
    st.markdown("### 📁 Upload Music")
    test_mp3 = st.file_uploader("", type=["mp3"])
    
    if test_mp3 is not None:
        st.markdown("### 🎧 Preview")
        st.audio(test_mp3)
        
        if st.button("🎯 Analyze Music", use_container_width=True):
            with st.spinner("🎵 Analyzing your music..."):
                result = process_single_file(test_mp3)
                
                # Display results in a structured format
                st.markdown("### 🎵 Genre Analysis")
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4 class='metric-title'>Predicted Genre</h4>
                        <h2 class='metric-value'>{result['genre'].title()}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Suggested genres
                suggested_genres = get_genre_recommendations(result['genre'])
                st.markdown("### 🎯 Suggested Genres")
                for genre in suggested_genres:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4 class='metric-title'>{genre.title()}</h4>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 📊 Quality Analysis")
                quality_metrics = result['quality_metrics']
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4 class='metric-title'>Audio Quality Score</h4>
                        <h2 class='metric-value'>{quality_metrics['snr_db']:.1f} dB</h2>
                        <p class='metric-title'>Signal-to-Noise Ratio</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4 class='metric-title'>Dynamic Range</h4>
                        <h2 class='metric-value'>{quality_metrics['dynamic_range_db']:.1f} dB</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4 class='metric-title'>Clipping</h4>
                        <h2 class='metric-value'>{quality_metrics['clipping_percentage']:.1f}%</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Musical Characteristics
                st.markdown("### 🎼 Musical Characteristics")
                features = result['cross_genre_features']
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4 class='metric-title'>Tempo</h4>
                        <h2 class='metric-value'>{features['tempo']:.1f} BPM</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4 class='metric-title'>Harmonic Complexity</h4>
                        <h2 class='metric-value'>{features['harmonic_complexity']:.2f}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4 class='metric-title'>Rhythm Complexity</h4>
                        <h2 class='metric-value'>{features['rhythm_complexity']:.2f}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Era prediction
                era = predict_year(result['audio_data'], result['sampling_rate'])
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4 class='metric-title'>Estimated Era</h4>
                        <h2 class='metric-value'>{era}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Generate PDF report
                pdf_bytes = generate_pdf_report(
                    result['audio_data'],
                    result['sampling_rate'],
                    result['genre'],
                    result['quality_metrics'],
                    result['cross_genre_features']
                )
                
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=pdf_bytes,
                    file_name="music_analysis_report.pdf",
                    mime="application/pdf"
                )

with tab2:
    st.markdown("### 📦 Batch Processing")
    uploaded_files = st.file_uploader("Upload multiple files", type=["mp3"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process All Files", use_container_width=True):
            with st.spinner("Processing files..."):
                results = process_multiple_files(uploaded_files)
                
                # Display results in a table
                st.markdown("### 📊 Batch Results")
                results_df = pd.DataFrame([
                    {
                        'File': file.name,
                        'Genre': result['genre'],
                        'Tempo': result['cross_genre_features']['tempo'],
                        'Quality Score': result['quality_metrics']['snr_db']
                    }
                    for file, result in zip(uploaded_files, results)
                ])
                st.dataframe(results_df)
                
                # Export results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="batch_results.csv",
                    mime="text/csv"
                )

with tab3:
    st.markdown("### 🎨 Music Visualization")
    vis_file = st.file_uploader("Upload a file for visualization", type=["mp3"])
    
    if vis_file is not None:
        st.audio(vis_file)
        
        if st.button("Generate Visualizations", use_container_width=True):
            with st.spinner("Generating visualizations..."):
                audio_data, sampling_rate = librosa.load(vis_file, sr=None)
                fig = create_visualizations(audio_data, sampling_rate)
                st.plotly_chart(fig, use_container_width=True)

# Add footer
st.markdown("""
    <div class='footer'>
        <p>Made by Mihir Pandya</p>
    </div>
""", unsafe_allow_html=True)
