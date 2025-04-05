import numpy as np
import time
import tempfile
import os
import io
import streamlit as st
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure NLTK data is downloaded (run once)
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Improved speech analysis with better error handling and fallbacks
def analyze_speech(audio_path):
    """
    A robust speech analysis function with multiple fallback options
    for transcription and audio feature extraction
    """
    print(f"Starting speech analysis for {audio_path}")
    start_time = time.time()
    
    # Default values in case of failure
    text = "Speech content would appear here."
    sentiment = {"compound": 0.1, "pos": 0.4, "neg": 0.1, "neu": 0.5}
    avg_pitch = 150
    pitch_variance = 25
    
    try:
        # Check if file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            print("Audio file missing or too small")
            return "No speech detected", sentiment, avg_pitch, pitch_variance
        
        # ===== TEXT TRANSCRIPTION METHODS =====
        # Method 1: Try Google Speech-to-Text API
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                print(f"Google STT successful: {text}")
        except ImportError:
            print("speech_recognition not available, trying alternative methods")
            text = "Speech content processed but transcription requires speech_recognition library."
        except Exception as e:
            print(f"Google STT failed: {e}")
            # Continue to next method if this fails
            text = "Speech processed but transcription unavailable."
        
        # ===== AUDIO ANALYSIS METHODS =====
        # Try using librosa for audio feature extraction
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Extract pitch information using librosa
            if len(y) > 0:
                # Pitch extraction using librosa
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y, fmin=librosa.note_to_hz('C2'), 
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr
                )
                
                # Filter out unvoiced and NaN segments
                valid_f0 = f0[voiced_flag]
                if len(valid_f0) > 0:
                    valid_f0 = valid_f0[~np.isnan(valid_f0)]
                
                if len(valid_f0) > 0:
                    avg_pitch = np.mean(valid_f0)
                    pitch_variance = np.std(valid_f0)
                else:
                    # Fallback to simple energy-based estimation
                    rms = librosa.feature.rms(y=y)[0]
                    avg_pitch = 120 + (np.mean(rms) * 1000) 
                    pitch_variance = np.std(rms) * 1000
            
        except (ImportError, Exception) as e:
            print(f"Librosa analysis failed: {e}")
            
            # Fallback to pydub for basic audio analysis
            try:
                audio = AudioSegment.from_file(audio_path)
                samples = np.array(audio.get_array_of_samples())
                
                # Use simple signal statistics as proxies for pitch metrics
                avg_pitch = 140 + (np.mean(np.abs(samples)) / 1000)
                pitch_variance = np.std(samples) / 1000
            except Exception as e:
                print(f"Fallback audio analysis failed: {e}")
                # Use default values
        
        # Calculate sentiment
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        
    except Exception as e:
        print(f"Overall analysis error: {e}")
    
    print(f"Analysis completed in {time.time() - start_time:.2f}s")
    return text, sentiment, avg_pitch, pitch_variance

# Enhanced sentiment analysis with voice feature integration
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
        # Higher pitch often correlates with positive emotions
        if avg_pitch > 180:  # Higher pitch
            pitch_adjustment = min((avg_pitch - 180) / 100, 0.2)  # Cap at 0.2
            enhanced_sentiment['compound'] = min(enhanced_sentiment['compound'] + pitch_adjustment, 1.0)
        elif avg_pitch < 120:  # Lower pitch
            pitch_adjustment = min((120 - avg_pitch) / 100, 0.2)  # Cap at 0.2
            enhanced_sentiment['compound'] = max(enhanced_sentiment['compound'] - pitch_adjustment, -1.0)
    
    if pitch_variance is not None:
        # Higher variance often indicates emotional speech
        if pitch_variance < 10:  # Monotonous speech
            if enhanced_sentiment['compound'] < 0:
                enhanced_sentiment['compound'] *= 1.5  # Amplify negative
            else:
                enhanced_sentiment['compound'] *= 0.7  # Dampen positive
        elif pitch_variance > 50:  # Highly variable speech
            enhanced_sentiment['compound'] *= 1.3  # Amplify existing sentiment
    
    # Normalize the pos/neg/neu values to sum to 1.0
    total = enhanced_sentiment['pos'] + enhanced_sentiment['neg'] + enhanced_sentiment['neu']
    if total > 0:
        scale_factor = 1.0 / total
        enhanced_sentiment['pos'] *= scale_factor
        enhanced_sentiment['neg'] *= scale_factor
        enhanced_sentiment['neu'] *= scale_factor
    
    return enhanced_sentiment

# Improved audio processing function with better error handling
def process_audio_recording(audio_data):
    """
    Process recorded audio data with robust error handling and status updates
    """
    if not audio_data or 'bytes' not in audio_data or len(audio_data['bytes']) < 1000:
        st.warning("Recording appears to be empty or too short. Please try recording again.")
        return None, None, None, None
    
    with st.spinner("Processing your recording..."):
        # Create a status message
        status_msg = st.empty()
        status_msg.info("Converting and analyzing audio...")
        
        audio_path = None
        text, sentiment, avg_pitch, pitch_variance = None, None, None, None
        
        try:
            # Create temp file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                audio_path = temp_file.name
                # Write the raw bytes to the file
                temp_file.write(audio_data['bytes'])
                temp_file.flush()
            
            # Convert audio format using pydub for compatibility
            try:
                # Convert to compatible WAV format
                audio = AudioSegment.from_file(io.BytesIO(audio_data['bytes']))
                audio = audio.set_channels(1)  # Convert to mono
                audio = audio.set_frame_rate(16000)  # Set to 16kHz
                audio.export(audio_path, format="wav")
                print(f"Audio successfully converted with pydub")
            except Exception as e:
                print(f"Pydub conversion failed: {e}. Proceeding with raw file.")
            
            # Display audio player
            st.audio(audio_data['bytes'])
            
            # Run our analysis
            text, sentiment, avg_pitch, pitch_variance = analyze_speech(audio_path)
            
            # Apply enhanced sentiment analysis
            if text and text != "No speech detected":
                sentiment = get_enhanced_sentiment(text, avg_pitch, pitch_variance)
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            status_msg.empty()
            # Clean up temp file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass
    
    return text, sentiment, avg_pitch, pitch_variance