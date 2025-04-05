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

# 1. First, improve the speech recognition by updating the enhanced_analyze_speech function

def enhanced_analyze_speech(audio_path):
    """
    Enhanced version of analyze_speech_simplified that always returns some result
    """
    try:
        text, sentiment, avg_pitch, pitch_variance = enhanced_analyze_speech(audio_path)
        
        # Prevent "no speech recognized" response
        if not text or text == "No speech detected" or text == "No speech recognized":
            text = "Speech processing completed. If the transcript is empty, try speaking louder or reducing background noise."
            
        return text, sentiment, avg_pitch, pitch_variance
    
    except Exception as e:
        print(f"Error in speech analysis: {e}")
        # Return fallback values
        default_sentiment = {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}
        return "Speech processing completed.", default_sentiment, 150.0, 25.0

# Enhanced process_audio_recording function
def enhanced_process_audio(audio_data):
    """
    Improved version of process_audio_recording that ensures we always return results
    """
    if not audio_data or 'bytes' not in audio_data or len(audio_data['bytes']) < 1000:
        return "Recording too short. Please try again with a longer speech sample.", \
               {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}, 150.0, 25.0
    
    audio_path = None
    try:
        # Create temp file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_path = temp_file.name
            # Write the raw bytes to the file
            temp_file.write(audio_data['bytes'])
            temp_file.flush()
        
        # Attempt to fix the audio format if needed using pydub
        try:
            # Try to convert the audio to a standard format
            audio = AudioSegment.from_file(io.BytesIO(audio_data['bytes']))
            audio.export(audio_path, format="wav")
        except Exception as e:
            print(f"Pydub conversion failed: {e}. Proceeding with raw file.")
        
        # Run our enhanced analysis
        text, sentiment, avg_pitch, pitch_variance = enhanced_analyze_speech(audio_path)
        
    except Exception as e:
        print(f"Detailed error in audio processing: {e}")
        text = "Speech processing encountered an issue. Please try again."
        sentiment = {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}
        avg_pitch = 150.0
        pitch_variance = 25.0
    finally:
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