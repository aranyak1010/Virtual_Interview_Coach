import speech_recognition as sr
from textblob import TextBlob
import numpy as np
import librosa
from nltk.sentiment import SentimentIntensityAnalyzer  # Ensure you import this

def analyze_speech(audio_path):
    recognizer = sr.Recognizer()
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    # Simple timing mechanism to track performance
    import time
    start_time = time.time()
    
    # Load the audio file - direct approach without extra processing
    try:
        with sr.AudioFile(audio_path) as source:
            # Minimal ambient noise adjustment
            recognizer.adjust_for_ambient_noise(source, duration=0.1)
            audio_data = recognizer.record(source)
            
            # Use only Google's service - fastest and most reliable
            try:
                text = recognizer.recognize_google(audio_data)
                print(f"Speech recognition completed in {time.time() - start_time:.2f}s")
            except sr.UnknownValueError:
                text = "No speech recognized"
                print(f"No speech recognized, elapsed time: {time.time() - start_time:.2f}s")
            except Exception as e:
                text = "No speech recognized"
                print(f"Recognition error: {str(e)}, elapsed time: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Audio file processing error: {str(e)}, elapsed time: {time.time() - start_time:.2f}s")
        return "Error processing audio", {"compound": 0, "pos": 0, "neg": 0, "neu": 1}, 0, 0

    # Simple sentiment analysis
    sentiment_time = time.time()
    sentiment = sentiment_analyzer.polarity_scores(text) if text and text != "No speech recognized" else {"compound": 0, "pos": 0, "neg": 0, "neu": 1}
    print(f"Sentiment analysis completed in {time.time() - sentiment_time:.2f}s")
    
    # Simplified pitch analysis - reduce complexity
    pitch_time = time.time()
    try:
        # Use a lower sample rate for faster processing
        y, sample_rate = librosa.load(audio_path, sr=8000)
        
        if len(y) > 0:
            # Use simpler pitch estimation - much faster
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate,
                frame_length=1024
            )
            
            # Filter out unvoiced segments
            pitch_values = f0[voiced_flag]
            
            if len(pitch_values) > 0:
                avg_pitch = np.mean(pitch_values)
                pitch_variance = np.std(pitch_values)
            else:
                avg_pitch = 120
                pitch_variance = 20
        else:
            avg_pitch = 120
            pitch_variance = 20
            
    except Exception as e:
        print(f"Pitch analysis error: {str(e)}, using default values")
        avg_pitch = 120
        pitch_variance = 20
    
    print(f"Pitch analysis completed in {time.time() - pitch_time:.2f}s")
    print(f"Total analysis time: {time.time() - start_time:.2f}s")
    
    return text, sentiment, avg_pitch, pitch_variance
