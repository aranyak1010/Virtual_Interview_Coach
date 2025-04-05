import speech_recognition as sr
import numpy as np
import librosa
from nltk.sentiment import SentimentIntensityAnalyzer
import time
import tempfile
import os
import io
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
import streamlit as st

def analyze_speech_simplified(audio_path):
    """Improved speech analysis with robust error handling and better audio processing"""
    print(f"Starting analysis for {audio_path}")
    start_time = time.time()
    
    # Default values
    text = "No speech recognized"
    sentiment = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    avg_pitch = 150
    pitch_variance = 25

    try:
        # Validate audio file
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            print("Invalid audio file")
            return "No speech detected", sentiment, avg_pitch, pitch_variance

        # Audio conversion pipeline
        cleaned_wav_path = audio_path + "_cleaned.wav"
        try:
            # FFmpeg conversion with proper parameters
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                "-loglevel", "error",  # Suppress verbose output
                cleaned_wav_path
            ], check=True, timeout=10)
            use_path = cleaned_wav_path
        except Exception as e:
            print(f"FFmpeg failed: {str(e)}")
            # Fallback to pydub conversion
            try:
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_channels(1).set_frame_rate(16000).normalize()
                audio.export(cleaned_wav_path, format="wav")
                use_path = cleaned_wav_path
            except Exception as e:
                print(f"Pydub conversion failed: {str(e)}")
                use_path = audio_path

        # Speech recognition setup
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 250  # Optimized for voice detection
        recognizer.pause_threshold = 1.0   # Longer pause detection

        try:
            with sr.AudioFile(use_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                
                # Primary recognition attempt
                try:
                    text = recognizer.recognize_google(audio_data, language="en-US", timeout=15)
                except sr.UnknownValueError:
                    # Fallback 1: Adjust parameters and retry
                    recognizer.energy_threshold = 200
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data, language="en-US")
                
        except sr.UnknownValueError:
            # Fallback 2: Use Sphinx with acoustic model adaptation
            try:
                text = recognizer.recognize_sphinx(audio_data)
            except:
                text = "No speech recognized"
        except Exception as e:
            print(f"Recognition error: {str(e)}")
            text = "Recognition service error"

        # Sentiment analysis
        if text and text not in ["No speech recognized", "Speech recognition failed"]:
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            
            # Pitch analysis
            try:
                y, sr = librosa.load(use_path, sr=16000)
                pitches = librosa.yin(y, fmin=50, fmax=400)
                valid_pitches = pitches[(pitches > 50) & (pitches < 400)]
                if len(valid_pitches) > 0:
                    avg_pitch = np.mean(valid_pitches)
                    pitch_variance = np.std(valid_pitches)
            except Exception as e:
                print(f"Pitch analysis error: {str(e)}")

        # Cleanup temporary files
        for f in [cleaned_wav_path, audio_path + "_pydub.wav"]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass

    except Exception as e:
        print(f"Critical error: {str(e)}")
    
    print(f"Analysis completed in {time.time() - start_time:.2f}s")
    return text, sentiment, avg_pitch, pitch_variance

def process_audio_recording(audio_data):
    """Robust audio processing pipeline with proper error handling"""
    if not audio_data or 'bytes' not in audio_data:
        return "No audio data", None, None, None
    
    audio_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(audio_data['bytes'])
            audio_path = tmpfile.name

        # Convert and normalize audio
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000).normalize()
        processed_path = audio_path + "_processed.wav"
        audio.export(processed_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        
        # Perform analysis
        return analyze_speech_simplified(processed_path)
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return "Processing failed", None, None, None
    finally:
        # Cleanup files
        for path in [audio_path, audio_path + "_processed.wav"]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except:
                pass

# Improve the sentiment analysis with a more sensitive algorithm
def get_enhanced_sentiment(text, avg_pitch=None, pitch_variance=None):
    """
    An improved sentiment analyzer with voice features integration
    and more accurate calibration
    """
    sia = SentimentIntensityAnalyzer()
    base_sentiment = sia.polarity_scores(text)
    
    # Create a copy to modify
    enhanced_sentiment = base_sentiment.copy()
    
    # Apply text-based enhancements
    
    # Look for emotion words and amplify their effect
    emotion_words = {
        'positive': ['happy', 'great', 'excellent', 'good', 'wonderful', 'amazing', 'love', 'awesome'],
        'negative': ['sad', 'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'upset', 'disappointed']
    }
    
    text_lower = text.lower()
    
    # Count emotion words
    pos_word_count = sum(1 for word in emotion_words['positive'] if word in text_lower)
    neg_word_count = sum(1 for word in emotion_words['negative'] if word in text_lower)
    
    # Apply stronger sentiment adjustments based on emotion words
    if pos_word_count > 0:
        sentiment_boost = min(0.1 * pos_word_count, 0.5)  # Cap at 0.5
        enhanced_sentiment['compound'] = min(enhanced_sentiment['compound'] + sentiment_boost, 1.0)
        enhanced_sentiment['pos'] += sentiment_boost
    
    if neg_word_count > 0:
        sentiment_penalty = min(0.1 * neg_word_count, 0.5)  # Cap at 0.5
        enhanced_sentiment['compound'] = max(enhanced_sentiment['compound'] - sentiment_penalty, -1.0)
        enhanced_sentiment['neg'] += sentiment_penalty
    
    # Increase sensitivity for neutral text
    if -0.1 < base_sentiment['compound'] < 0.1:
        # Apply more aggressive scaling to push away from neutral
        if base_sentiment['compound'] > 0:
            enhanced_sentiment['compound'] = enhanced_sentiment['compound'] * 2.5
        elif base_sentiment['compound'] < 0:
            enhanced_sentiment['compound'] = enhanced_sentiment['compound'] * 2.5
    
    # Apply voice feature adjustments if available
    if avg_pitch is not None:
        # Higher pitch often correlates with positive emotions (happiness)
        # Lower pitch often correlates with negative emotions (sadness)
        # Adjust sentiment based on deviation from neutral pitch (around 150Hz)
        if avg_pitch > 180:  # Higher pitch
            pitch_adjustment = min((avg_pitch - 180) / 100, 0.2)  # Cap at 0.2
            enhanced_sentiment['compound'] = min(enhanced_sentiment['compound'] + pitch_adjustment, 1.0)
        elif avg_pitch < 120:  # Lower pitch
            pitch_adjustment = min((120 - avg_pitch) / 100, 0.2)  # Cap at 0.2
            enhanced_sentiment['compound'] = max(enhanced_sentiment['compound'] - pitch_adjustment, -1.0)
    
    if pitch_variance is not None:
        # Higher variance often indicates emotional speech (either positive or negative)
        # Low variance suggests monotony, which might indicate neutrality or depression
        if pitch_variance < 10:  # Monotonous speech
            # If already negative, make more negative (potential depression indicator)
            if enhanced_sentiment['compound'] < 0:
                enhanced_sentiment['compound'] *= 1.5  # Amplify negative
            else:
                # For positive or neutral, push toward neutral
                enhanced_sentiment['compound'] *= 0.7  # Dampen positive
        elif pitch_variance > 50:  # Highly variable speech - amplify existing sentiment
            enhanced_sentiment['compound'] *= 1.3  # Amplify whatever sentiment exists
    
    # Normalize the pos/neg/neu values to sum to 1.0
    total = enhanced_sentiment['pos'] + enhanced_sentiment['neg'] + enhanced_sentiment['neu']
    if total > 0:
        scale_factor = 1.0 / total
        enhanced_sentiment['pos'] *= scale_factor
        enhanced_sentiment['neg'] *= scale_factor
        enhanced_sentiment['neu'] *= scale_factor
    
    return enhanced_sentiment
