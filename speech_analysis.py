import speech_recognition as sr
import numpy as np
import librosa
from nltk.sentiment import SentimentIntensityAnalyzer
import time

def analyze_speech(audio_path):
    recognizer = sr.Recognizer()
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    start_time = time.time()  # Start timing the entire function

    try:
        with sr.AudioFile(audio_path) as source:
            # Shorter ambient noise adjustment for faster processing
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)

            # Debugging output
            print(f"Length of audio data: {len(audio_data.frame_data)}")
            
            try:
                # Faster speech recognition with timeout
                text = recognizer.recognize_google(audio_data, timeout=10)
                print("Speech recognized:", text)
            except sr.UnknownValueError:
                text = "No speech recognized"
                print("No speech recognized")
            except Exception as e:
                text = "No speech recognized"
                print(f"Recognition error: {str(e)}")

    except Exception as e:
        print(f"Audio file processing error: {str(e)}")
        return "Error processing audio", {"compound": 0, "pos": 0, "neg": 0, "neu": 1}, 0, 0

    # Simple sentiment analysis
    sentiment_time = time.time()
    sentiment = sentiment_analyzer.polarity_scores(text) if text and text != "No speech recognized" else {"compound": 0, "pos": 0, "neg": 0, "neu": 1}
    print(f"Sentiment analysis completed in {time.time() - sentiment_time:.2f}s")
    
    # Simplified pitch analysis with lower sample rate for faster processing
    pitch_time = time.time()
    try:
        # Use a much lower sample rate for faster processing
        y, sample_rate = librosa.load(audio_path, sr=8000, duration=10)  # Limit to 10 seconds for speed
        
        if len(y) > 0 and np.mean(np.abs(y)) > 0.01:  # Check if audio has actual content
            # Use simpler and faster pitch estimation
            pitches, magnitudes = librosa.piptrack(
                y=y, 
                sr=sample_rate,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                n_fft=1024,  # Smaller FFT window
                hop_length=512  # Larger hop for speed
            )
            
            # Get pitches with significant magnitude
            pitch_values = []
            for i in range(min(10, magnitudes.shape[1])):  # Limit columns for speed
                index = magnitudes[:,i].argmax()
                pitch = pitches[index,i]
                if pitch > 0:
                    pitch_values.append(pitch)
            
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
