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

# 1. First, improve the speech recognition by updating the analyze_speech_simplified function

def analyze_speech_simplified(audio_path):
    """
    Improved speech analysis function with better error handling and recognition
    """
    print(f"Starting improved speech analysis for {audio_path}")
    start_time = time.time()
    
    # Default values in case of failure
    text = "No speech recognized"
    sentiment = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    avg_pitch = 150
    pitch_variance = 25
    
    try:
        # Check if file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            print("Audio file missing or too small")
            return "No speech detected", sentiment, avg_pitch, pitch_variance
            
        # Use Google Speech-to-Text API with improved parameters
        try:
            recognizer = sr.Recognizer()
            # Add adjustments to the recognizer for better performance
            recognizer.energy_threshold = 300  # Default is 300
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8  # Default is 0.8
            
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise with longer duration
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                
                # Try with Google first with increased timeout
                try:
                    text = recognizer.recognize_google(audio_data, language="en-US", show_all=False, timeout=10)
                    print(f"Google STT successful: {text}")
                except sr.UnknownValueError:
                    print("Google STT couldn't understand audio")
                    text = "No speech recognized"
                except sr.RequestError as e:
                    print(f"Google STT service error: {e}")
                    # Try with Sphinx as fallback if Google fails
                    try:
                        text = recognizer.recognize_sphinx(audio_data)
                        print(f"Sphinx STT fallback successful: {text}")
                    except:
                        text = "Speech recognition service unavailable"
        except Exception as e:
            print(f"Speech recognition failed: {e}")
            text = "Speech recognition failed, please try again"

        # Calculate sentiment only if we have text
        if text and text != "No speech recognized" and text != "Speech recognition failed, please try again":
            # Use enhanced sentiment analysis
            sentiment = get_enhanced_sentiment(text, avg_pitch, pitch_variance)
        
        # Generate pitch metrics using librosa
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Extract pitch using a more reliable method
            if len(y) > 0:
                # Use librosa's pitch tracking
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitches = pitches[magnitudes > np.median(magnitudes)]
                
                # Filter out extreme values
                if len(pitches) > 0:
                    filtered_pitches = pitches[(pitches > 50) & (pitches < 400)]
                    if len(filtered_pitches) > 0:
                        avg_pitch = np.mean(filtered_pitches)
                        pitch_variance = np.std(filtered_pitches)
                    else:
                        # Fallback if filtering removed all values
                        avg_pitch = 150
                        pitch_variance = 25
            
        except Exception as lib_error:
            print(f"Pitch analysis failed: {lib_error}")
            # Keep default values if analysis fails
            
    except Exception as e:
        print(f"Overall analysis error: {e}")
    
    print(f"Analysis completed in {time.time() - start_time:.2f}s")
    return text, sentiment, avg_pitch, pitch_variance

# 2. Improve the audio processing pipeline
def process_audio_recording(audio_data):
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
            # Create temp file for audio with a more reliable approach
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                audio_path = temp_file.name
            
            # Use pydub to handle audio format conversion more reliably
            try:
                # Convert from webm to wav using pydub
                audio = AudioSegment.from_file(io.BytesIO(audio_data['bytes']), format="webm")
                # Normalize audio to improve speech recognition
                normalized_audio = audio.normalize(headroom=4.0)
                # Export as WAV with parameters optimized for speech recognition
                normalized_audio.export(
                    audio_path, 
                    format="wav",
                    parameters=["-ac", "1", "-ar", "16000"]  # Mono, 16kHz
                )
                print(f"Audio successfully converted and normalized")
            except Exception as e:
                print(f"Pydub conversion failed: {e}. Attempting direct write.")
                # Fallback to direct write if pydub fails
                with open(audio_path, 'wb') as f:
                    f.write(audio_data['bytes'])
            
            # Display audio player
            st.audio(audio_data['bytes'])
            
            # Run our improved analysis
            text, sentiment, avg_pitch, pitch_variance = analyze_speech_simplified(audio_path)
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        finally:
            status_msg.empty()
            # Clean up temp file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass
    
    return text, sentiment, avg_pitch, pitch_variance

# 3. Improve the sentiment analysis with a more sensitive algorithm
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
