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
    """
    Speech analysis function with more robust transcription handling
    """
    print(f"Starting speech analysis with improved transcription for {audio_path}")
    start_time = time.time()
    
    # Default values in case of failure
    text = "No speech recognized"
    sentiment = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    avg_pitch = 150
    pitch_variance = 25
    
    try:
        # Check if file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            print(f"Audio file missing or too small: {audio_path}, size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'file not found'}")
            return "No speech detected", sentiment, avg_pitch, pitch_variance
        
        # DEBUG: Print file info
        print(f"Audio file size: {os.path.getsize(audio_path)} bytes")
        
        # First attempt: Use the recognizer directly with explicit WAV conversion
        try:
            # Convert to proper WAV format if needed
            import wave
            import subprocess
            import soundfile as sf
            
            # Check if it's a valid audio file and get info
            try:
                info = sf.info(audio_path)
                print(f"SoundFile info: {info.samplerate}Hz, {info.channels} channels, {info.format}")
            except Exception as sf_error:
                print(f"SoundFile info error: {sf_error}")
            
            # Create a clean WAV file that should be compatible with recognizer
            cleaned_wav_path = audio_path + "_cleaned.wav"
            try:
                # Try using ffmpeg for reliable conversion
                subprocess.run([
                    "ffmpeg", "-y", "-i", audio_path, 
                    "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                    cleaned_wav_path
                ], check=True, capture_output=True)
                print("FFmpeg conversion successful")
                use_path = cleaned_wav_path
            except Exception as ffmpeg_error:
                print(f"FFmpeg conversion error: {ffmpeg_error}")
                # Fallback to direct soundfile conversion
                try:
                    data, samplerate = sf.read(audio_path)
                    sf.write(cleaned_wav_path, data, 16000, subtype='PCM_16')
                    print("SoundFile conversion successful")
                    use_path = cleaned_wav_path
                except Exception as sf_write_error:
                    print(f"SoundFile conversion error: {sf_write_error}")
                    use_path = audio_path  # Use original if conversion fails
            
            recognizer = sr.Recognizer()
            # Set recognizer properties for better recognition
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.dynamic_energy_adjustment_damping = 0.15
            recognizer.dynamic_energy_ratio = 1.5
            recognizer.pause_threshold = 0.8
            recognizer.non_speaking_duration = 0.5
            
            # First attempt with Google Speech Recognition
            print(f"Opening audio file for recognition: {use_path}")
            with sr.AudioFile(use_path) as source:
                # Adjust for ambient noise with optimal duration
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Recording from audio file...")
                audio_data = recognizer.record(source)
                print(f"Audio duration: approximately {len(audio_data.frame_data)/(audio_data.sample_rate*audio_data.sample_width)} seconds")
                
                # Try Google STT with increased timeout
                try:
                    print("Attempting Google STT...")
                    text = recognizer.recognize_google(audio_data, language="en-US", timeout=15)
                    print(f"Google STT successful: '{text}'")
                except sr.UnknownValueError:
                    print("Google STT couldn't understand audio")
                    # Try again with more aggressive noise reduction
                    try:
                        print("Retrying with adjusted audio...")
                        # Try different energy threshold
                        recognizer.energy_threshold = 50  # Lower threshold
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data, language="en-US", timeout=15)
                        print(f"Google STT retry successful: '{text}'")
                    except:
                        # Try with Sphinx as fallback
                        try:
                            print("Attempting Sphinx fallback...")
                            text = recognizer.recognize_sphinx(audio_data)
                            print(f"Sphinx STT successful: '{text}'")
                        except:
                            text = "No speech recognized"
                except sr.RequestError as e:
                    print(f"Google STT service error: {e}")
                    # Try with Sphinx as fallback
                    try:
                        print("Attempting Sphinx fallback due to request error...")
                        text = recognizer.recognize_sphinx(audio_data)
                        print(f"Sphinx STT successful: '{text}'")
                    except:
                        text = "Speech recognition service unavailable"
        except Exception as rec_error:
            print(f"Recognition error: {rec_error}")
            
            # Second attempt: Try using pydub to preprocess the audio
            try:
                print("Attempting alternate approach with pydub...")
                from pydub import AudioSegment
                
                # Load and normalize the audio
                audio = AudioSegment.from_file(audio_path)
                # Convert to mono and set sample rate
                audio = audio.set_channels(1).set_frame_rate(16000)
                # Normalize audio
                audio = audio.normalize()
                # Export as WAV
                pydub_wav_path = audio_path + "_pydub.wav"
                audio.export(pydub_wav_path, format="wav")
                
                # Try recognition again
                recognizer = sr.Recognizer()
                with sr.AudioFile(pydub_wav_path) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data, language="en-US")
                        print(f"Pydub+Google successful: '{text}'")
                    except:
                        try:
                            text = recognizer.recognize_sphinx(audio_data)
                            print(f"Pydub+Sphinx successful: '{text}'")
                        except:
                            text = "Speech recognition failed"
            except Exception as pydub_error:
                print(f"Pydub approach error: {pydub_error}")
                text = "Speech recognition failed"
                
        # Continue with sentiment and pitch analysis (unchanged)
        if text and text != "No speech recognized" and text != "Speech recognition failed":
            # Calculate sentiment
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            
            # Calculate pitch metrics using librosa
            try:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                
                # Basic pitch analysis
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                # Filter out silent parts
                pitches = pitches[magnitudes > 0.05]
                if len(pitches) > 0:
                    # Filter out unlikely pitch values for human speech
                    filtered_pitches = pitches[(pitches > 50) & (pitches < 400)]
                    if len(filtered_pitches) > 0:
                        avg_pitch = np.mean(filtered_pitches)
                        pitch_variance = np.std(filtered_pitches)
            except Exception as pitch_error:
                print(f"Pitch analysis error: {pitch_error}")
        
        # Clean up temp files
        for temp_path in [cleaned_wav_path, audio_path + "_pydub.wav"]:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
    except Exception as e:
        print(f"Overall analysis error: {e}")
    
    print(f"Analysis completed in {time.time() - start_time:.2f}s")
    return text, sentiment, avg_pitch, pitch_variance

# Improved audio processing for better transcription
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
            # Create temp file with a specific name pattern for easier debugging
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = os.path.join(tempfile.gettempdir(), f"streamlit_audio_{timestamp}.wav")
            
            # Save the raw audio data first
            with open(audio_path, 'wb') as f:
                f.write(audio_data['bytes'])
            
            print(f"Raw audio saved to {audio_path}")
            
            # # Try to extract audio format information
            # try:
            #     # Try to determine the actual format
                
            #     file_type = magic.from_file(audio_path)
            #     print(f"File type detected: {file_type}")
            # except:
            #     print("Could not detect file type with magic library")
            
            # Convert audio using pydub with extra care
            try:
                from pydub import AudioSegment
                
                # Try to load with explicit format
                if 'format' in audio_data and audio_data['format']:
                    print(f"Using format from audio_data: {audio_data['format']}")
                    audio_format = audio_data['format']
                else:
                    print("No format specified, assuming webm")
                    audio_format = "webm"
                
                # Try different approaches to load the audio
                try:
                    audio = AudioSegment.from_file(audio_path, format=audio_format)
                except:
                    try:
                        audio = AudioSegment.from_file(audio_path)  # Let pydub detect format
                    except:
                        audio = AudioSegment.from_file(io.BytesIO(audio_data['bytes']), format=audio_format)
                
                # Process audio for better recognition
                audio = audio.set_channels(1)  # Convert to mono
                audio = audio.set_frame_rate(16000)  # Set to 16kHz
                audio = audio.normalize(headroom=4.0)  # Normalize volume
                
                # Save the processed audio
                processed_audio_path = audio_path + "_processed.wav"
                audio.export(
                    processed_audio_path, 
                    format="wav",
                    parameters=["-ac", "1", "-ar", "16000"]  # Ensure mono, 16kHz
                )
                print(f"Processed audio saved to {processed_audio_path}")
                
                # Use the processed audio file for analysis
                audio_path = processed_audio_path
            except Exception as conversion_error:
                print(f"Audio conversion error: {conversion_error}")
                # Continue with the raw file
            
            # Display the audio player
            st.audio(audio_data['bytes'])
            
            # Run our improved analysis
            text, sentiment, avg_pitch, pitch_variance = analyze_speech_simplified(audio_path)
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            status_msg.empty()
            # Clean up temp file
            for path in [audio_path, audio_path + "_processed.wav"]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
    
    return text, sentiment, avg_pitch, pitch_variance

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
