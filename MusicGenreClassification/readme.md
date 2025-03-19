🎵 Music Genre Classification App

This project is a Music Genre Classification App built with Python and Streamlit. It classifies audio files into different music genres using machine learning and deep learning techniques. The app allows users to upload .mp3 files and get instant predictions along with model insights.

🚀 Features

✅ Upload and classify your own music files
✅ Predict music genres using a trained CNN model
✅ Visualize training history (accuracy & loss graphs)
✅ Simple and interactive Streamlit web app
✅ Clean modular code structure for easy maintenance

This project utilizes the GTZAN Music Genre Dataset, which was obtained from Kaggle. The dataset includes audio tracks labeled across 10 genres, providing a solid foundation for building and training the classification model.

## 📁 Project Structure
MusicGenreClassification/
├── data/
│   └── genres_original/        # Dataset (organized by genre folders)
│       ├── blues/
│       ├── classical/
│       ├── country/
│       ├── disco/
│       ├── hiphop/
│       ├── jazz/
│       ├── metal/
│       ├── pop/
│       ├── reggae/
│       └── rock/
├── notebooks/
│   └── music.ipynb             # Jupyter Notebook for EDA & experimentation
├── app/
│   ├── Music_Genre_App.py      # Streamlit app file
│   └── assets/                 # Images, audio samples, etc.
│       └── image.png
├── models/
│   └── Trained_model.keras     # Saved trained model
├── results/
│   └── training_hist.json      # Training history (accuracy, loss)
├── .gitignore
├── README.md
└── requirements.txt


🧠 Model Details

    Dataset: GTZAN Music Genre Dataset (or your custom dataset)
    Genres Covered:
        Blues
        Classical
        Country
        Disco
        Hip-Hop
        Jazz
        Metal
        Pop
        Reggae
        Rock
    Model Type: Convolutional Neural Network (CNN)
    File: /models/Trained_model.keras
    Training Metrics: Accuracy and loss graphs are stored in /results/training_hist.json

📊 Results & Visualizations

    Training history: Visualized in the Streamlit app
    Accuracy and loss over time plotted using Matplotlib/Seaborn
    Predicted genre displayed instantly after upload

📂 Dataset Info

    Folder structure: /data/genres_original/
    Each genre has its own folder containing .wav or .au audio files
    Dataset used for model training and testing
    You can replace this with your own dataset following the same structure

🔨 Tech Stack

    Python 3.x
    Streamlit
    TensorFlow / Keras
    Librosa (audio feature extraction)
    NumPy, Pandas
    Matplotlib / Seaborn
    Scikit-learn

✨ Future Improvements

    Add real-time audio recording and classification
    Expand the dataset with more genres and samples
    Deploy the app on Streamlit Cloud, Hugging Face Spaces, or Heroku
    Add user authentication and history tracking
    Improve accuracy with advanced deep learning models like CNN-LSTM or Transformers
