import speech_recognition as sr
import librosa
import numpy as np
import soundfile as sf
import tempfile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        print("Transcribing audio...")
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("Transcription:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return ""
    except sr.RequestError:
        print("Error with Google Speech-to-Text API.")
        return ""

def analyze_sentiment(text):
    if text:
        sentiment = sia.polarity_scores(text)
        print("Sentiment Analysis:", sentiment)
        return sentiment
    return {}

def analyze_tone(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    
    # Compute average pitch excluding NaN values
    avg_pitch = np.nanmean(pitch) if np.any(~np.isnan(pitch)) else 0.0
    
    # Compute pitch variability
    pitch_variability = np.nanstd(pitch) if np.any(~np.isnan(pitch)) else 0.0
    
    print(f"Average Pitch: {avg_pitch:.2f} Hz")
    print(f"Pitch Variability: {pitch_variability:.2f} Hz")
    return avg_pitch, pitch_variability

def record_audio_realtime():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording... Speak now!")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print("Live Transcription:", text)
        
        # Convert raw audio bytes to NumPy array
        audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
        
        # Save to temporary WAV file for pitch analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name
            sf.write(temp_audio_path, audio_data, samplerate=16000)
        
        # Analyze pitch from recorded audio
        avg_pitch, pitch_variability = analyze_tone(temp_audio_path)
        print(f"Final Average Pitch (Real-time): {avg_pitch:.2f} Hz")
        
        # Store the audio data for potential playback
        record_audio_realtime.audio_data = {'bytes': audio.get_wav_data()}
        
        # Return all three expected values
        return text, avg_pitch, pitch_variability
        
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        # Return default values when speech isn't recognized
        return "", 0.0, 0.0
    except sr.RequestError:
        print("Error with Google Speech-to-Text API.")
        # Return default values on API error
        return "", 0.0, 0.0
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Return default values on any other error
        return "", 0.0, 0.0

# Example usage
if __name__ == "__main__":
    mode = input("Choose mode: (1) Upload Audio File, (2) Real-time Recording: ")
    if mode == "1":
        audio_file = "sample_audio.wav"  # Change to an actual audio file
        text = transcribe_audio(audio_file)
        avg_pitch, pitch_variability = analyze_tone(audio_file)
        print(f"Final Average Pitch: {avg_pitch:.2f} Hz")
    elif mode == "2":
        text = record_audio_realtime()
    else:
        print("Invalid choice. Exiting.")
        exit()
    
    sentiment = analyze_sentiment(text)
    print("Final Analysis:")
    print(f"Speech Sentiment: {sentiment}")
