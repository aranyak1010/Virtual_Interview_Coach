import streamlit as st
import speech_recognition as sr
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import io
from pydub import AudioSegment

# Initialize NLTK with better error handling
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Ensure NLTK resources are downloaded (only needed once)
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    
    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()
    NLTK_AVAILABLE = True
except Exception as e:
    print(f"Error initializing NLTK: {e}")
    NLTK_AVAILABLE = False

def analyze_sentiment(text):
    """
    Analyze sentiment of transcribed text
    Works with or without NLTK available
    """
    if not text or text == "Speech not recognized. Please try again with clearer speech.":
        return {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}
    
    if NLTK_AVAILABLE:
        try:
            sentiment = sia.polarity_scores(text)
            print("Sentiment Analysis:", sentiment)
            
            # Apply enhanced sensitivity to make results less neutral
            compound = sentiment['compound']
            
            # Exponential amplification of the compound score while preserving sign
            enhanced_compound = compound * (1.5 + abs(compound))
            if enhanced_compound > 1.0:
                enhanced_compound = 1.0
            elif enhanced_compound < -1.0:
                enhanced_compound = -1.0
            
            # Amplify pos/neg scores to reduce neutrality
            pos = sentiment['pos'] * 1.25
            neg = sentiment['neg'] * 1.25
            
            # Ensure pos + neg + neu = 1.0
            total = pos + neg
            if total > 1.0:
                pos = pos / total
                neg = neg / total
                neu = 0.0
            else:
                neu = 1.0 - (pos + neg)
            
            # Create enhanced sentiment
            enhanced_sentiment = {
                "compound": enhanced_compound,
                "pos": pos,
                "neg": neg,
                "neu": neu
            }
            
            return enhanced_sentiment
        except Exception as e:
            print(f"Error analyzing sentiment with NLTK: {e}")
    
    # Fallback to TextBlob if NLTK is not available or fails
    try:
        from textblob import TextBlob
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        # Convert polarity to VADER-like format
        return {
            "compound": polarity,
            "pos": max(0, polarity),
            "neg": max(0, -polarity),
            "neu": 1.0 - abs(polarity)
        }
    except Exception:
        # Ultimate fallback if all sentiment analysis fails
        return {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}

# The rest of your functions remain the same
def enhanced_process_audio(audio_data):
    """
    Process audio data from streamlit-mic-recorder for transcription and analysis
    """
    if not audio_data or 'bytes' not in audio_data or len(audio_data['bytes']) < 1000:
        return "Recording too short. Please try again with a longer speech sample.", \
               {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}, 150.0, 25.0
    
    # Direct in-memory approach - no temporary files for WebM
    try:
        # Create a BytesIO object with the audio bytes
        audio_bytes = io.BytesIO(audio_data['bytes'])
        
        # Create a temporary WAV file with a guaranteed path
        fd, wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)  # Close the file descriptor but keep the file
        
        try:
            # Convert WebM to WAV using pydub directly from bytes
            audio = AudioSegment.from_file(audio_bytes, format="webm")
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set to 16kHz for better speech recognition
            audio.export(wav_path, format="wav")
            
            print(f"Audio converted: {audio.duration_seconds:.2f}s, {audio.channels} channels, {audio.frame_rate}Hz")
        except Exception as e:
            print(f"Audio conversion error: {e}")
            return f"Audio conversion failed: {str(e)}", {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}, 150.0, 25.0
        
        # Verify the file exists and has content
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 100:
            return "WAV file creation failed or file is too small", {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}, 150.0, 25.0
        
        # Use the working transcription function from microphone.py
        text = transcribe_audio(wav_path)
        
        # Use the working sentiment analysis function from microphone.py
        sentiment = analyze_sentiment(text)
        
        # Use the working tone analysis function from microphone.py
        avg_pitch, pitch_variance = analyze_tone(wav_path)
        
        return text, sentiment, avg_pitch, pitch_variance
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing audio: {str(e)}", {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}, 150.0, 25.0
    finally:
        # Clean up temp WAV file
        if 'wav_path' in locals() and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as e:
                print(f"Error cleaning up {wav_path}: {e}")

def transcribe_audio(audio_path):
    """
    Transcribe audio using Google Speech Recognition
    Direct integration from the working microphone.py
    """
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
        return "Speech not recognized. Please try again with clearer speech."
    except sr.RequestError:
        print("Error with Google Speech-to-Text API.")
        return "Speech service unavailable. Please try again later."

def analyze_tone(audio_path):
    """
    Analyze pitch and other vocal characteristics
    Direct integration from the working microphone.py
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitch = librosa.yin(y, fmin=50, fmax=300)
        
        # Compute average pitch excluding NaN values
        avg_pitch = np.nanmean(pitch) if np.any(~np.isnan(pitch)) else 0.0
        
        # Compute pitch variability
        pitch_variability = np.nanstd(pitch) if np.any(~np.isnan(pitch)) else 0.0
        
        print(f"Average Pitch: {avg_pitch:.2f} Hz")
        print(f"Pitch Variability: {pitch_variability:.2f} Hz")
        return avg_pitch, pitch_variability
    except Exception as e:
        print(f"Error analyzing tone: {e}")
        return 150.0, 25.0

def record_audio_realtime():
    """
    Record audio in real-time using microphone
    Direct integration from the working microphone.py
    """
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
        
        return text, avg_pitch, pitch_variability
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return "", 0.0, 0.0
    except sr.RequestError:
        print("Error with Google Speech-to-Text API.")
        return "", 0.0, 0.0