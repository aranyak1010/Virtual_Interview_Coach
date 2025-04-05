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

# DEBUG FLAG - Set to True to help troubleshoot issues
DEBUG = True

# Fix 1: Improved audio format handling with more explicit conversions
def convert_audio_for_recognition(input_path):
    """Convert audio to a format known to work well with speech recognition"""
    try:
        # Load audio using pydub
        audio = AudioSegment.from_file(input_path)
        
        # Convert to appropriate format for speech recognition
        audio = audio.set_channels(1)           # Mono
        audio = audio.set_frame_rate(16000)     # 16kHz sample rate
        audio = audio.set_sample_width(2)       # 16-bit depth
        
        # Export to a new temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_path = temp_file.name
        temp_file.close()
        
        # Export with specific parameters known to work with speech_recognition
        audio.export(
            output_path, 
            format="wav",
            parameters=["-ac", "1", "-ar", "16000"]
        )
        
        if DEBUG:
            print(f"Audio converted: {len(audio)/1000:.2f}s duration, {audio.channels} channels, {audio.frame_rate}Hz")
            
        return output_path
    except Exception as e:
        if DEBUG:
            print(f"Audio conversion failed: {e}")
        return input_path  # Return original path if conversion fails

# Fix 2: More robust speech detection with multiple attempts and adjustable parameters
def analyze_speech(audio_path):
    """
    A robustified speech analysis function with multiple attempts
    """
    print(f"Starting speech analysis for {audio_path}")
    start_time = time.time()
    
    # Default values
    text = "No speech detected"  # Set this as default to track if any method succeeds
    sentiment = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    avg_pitch = 150
    pitch_variance = 25
    
    # Check if file exists and has content
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
        print("Audio file missing or too small")
        return text, sentiment, avg_pitch, pitch_variance
    
    # Fix 3: Create a processed version of the audio optimized for speech recognition
    processed_audio_path = convert_audio_for_recognition(audio_path)
    
    # ===== TRANSCRIPTION APPROACHES =====
    # We'll try multiple approaches and use the first one that works
    
    # Approach 1: SpeechRecognition with Google
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        # Fix 4: Configure recognizer for better results
        recognizer.energy_threshold = 200  # Lower energy threshold (even more sensitive)
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.0  # Be more tolerant of pauses
        recognizer.operation_timeout = 5  # Add operation timeout
        
        with sr.AudioFile(processed_audio_path) as source:
            # Fix 5: Adjust for ambient noise for a shorter duration
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            
            # Fix 6: Record with longer timeout
            audio_data = recognizer.record(source)
            
            # Fix 7: Attempt recognition with multiple services
            # First try Google
            try:
                text = recognizer.recognize_google(
                    audio_data,
                    language="en-US",  # Explicitly specify language
                    show_all=False     # Return only the most likely result
                )
                
                if text and text.strip():
                    print(f"Google STT successful: '{text}'")
                else:
                    text = "No speech detected"  # Reset if empty string returned
            except Exception as e:
                print(f"Google STT failed: {e}")
                # If Google fails, try Sphinx as fallback
                try:
                    # Only attempt if sphinx is installed
                    text = recognizer.recognize_sphinx(audio_data)
                    if text and text.strip():
                        print(f"Sphinx STT successful: '{text}'")
                    else:
                        text = "No speech detected"
                except (ImportError, sr.UnknownValueError) as e:
                    print(f"Sphinx STT failed: {e}")
                    text = "No speech detected"
                
    except ImportError as ie:
        print(f"speech_recognition import error: {ie}")
        # Ensure SpeechRecognition is installed
        try:
            import pip
            pip.main(['install', 'SpeechRecognition'])
            print("Installed SpeechRecognition package")
        except:
            print("Failed to install SpeechRecognition")
    except sr.UnknownValueError:
        if DEBUG:
            print("STT couldn't understand audio")
    except Exception as e:
        if DEBUG:
            print(f"STT error: {e}")
    
    # Fix 8: Fallback - If transcription services don't work, try a simpler approach
    if text == "No speech detected":
        try:
            # Check if the audio has enough variation to likely contain speech
            audio = AudioSegment.from_file(processed_audio_path)
            samples = np.array(audio.get_array_of_samples())
            
            # Enhanced heuristic - check for likely speech patterns
            rms = np.sqrt(np.mean(samples**2))
            
            # Check for audio energy and variation
            if rms > 50:  # Lower threshold to detect quieter speech
                # Check for variations in audio that suggest speech
                chunks = np.array_split(samples, min(10, len(samples)))
                chunk_means = [np.mean(abs(chunk)) for chunk in chunks if len(chunk) > 0]
                
                # If there's enough variation between quiet and loud parts
                if len(chunk_means) > 1 and max(chunk_means) / (min(chunk_means) + 0.01) > 2:
                    text = "Speech detected but transcription failed. Please try speaking louder and more clearly."
                    print(f"Basic audio check: Speech pattern detected (RMS: {rms})")
        except Exception as e:
            if DEBUG:
                print(f"Basic audio check failed: {e}")
    
    # ===== AUDIO ANALYSIS =====
    try:
        # Extract audio features using pydub (works without librosa)
        audio = AudioSegment.from_file(processed_audio_path)
        samples = np.array(audio.get_array_of_samples())
        
        # Simple statistics-based estimations
        if len(samples) > 0:
            # Normalize samples
            samples = samples / (2**15) if samples.dtype == np.int16 else samples / (2**31)
            
            # Estimate pitch - this is a simple estimation, not accurate pitch detection
            # Real pitch detection requires more complex algorithms
            chunk_size = int(audio.frame_rate * 0.03)  # 30ms chunks
            chunks = [samples[i:i+chunk_size] for i in range(0, len(samples), chunk_size) if i+chunk_size < len(samples)]
            
            if len(chunks) > 0:
                # Calculate zero-crossing rate for each chunk (correlates with pitch)
                zcrs = []
                for chunk in chunks:
                    # Zero crossing rate
                    zcr = sum(abs(np.diff(np.signbit(chunk).astype(int)))) / (2 * len(chunk))
                    zcrs.append(zcr * audio.frame_rate)
                
                # Filter out unreasonably high/low values (simple noise filtering)
                filtered_zcrs = [z for z in zcrs if 70 < z < 400]  # Human speech range approx.
                
                if filtered_zcrs:
                    avg_pitch = np.mean(filtered_zcrs)
                    pitch_variance = np.std(filtered_zcrs)
                else:
                    # Default to typical male/female average if we couldn't detect
                    avg_pitch = 150
                    pitch_variance = 30
            else:
                avg_pitch = 150
                pitch_variance = 30
    except Exception as e:
        if DEBUG:
            print(f"Audio feature extraction failed: {e}")
    
    # If we have speech text, calculate sentiment
    if text and text != "No speech detected":
        try:
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
        except Exception as e:
            if DEBUG:
                print(f"Sentiment analysis failed: {e}")
    
    # Clean up temporary files
    if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
        try:
            os.unlink(processed_audio_path)
        except:
            pass
    
    print(f"Analysis completed in {time.time() - start_time:.2f}s")
    return text, sentiment, avg_pitch, pitch_variance

# Same function as before, just reusing for completeness
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
    
    # Apply adjustments based on emotion words
    if pos_word_count > 0:
        sentiment_boost = min(0.1 * pos_word_count, 0.5)  # Cap at 0.5
        enhanced_sentiment['compound'] = min(enhanced_sentiment['compound'] + sentiment_boost, 1.0)
        enhanced_sentiment['pos'] += sentiment_boost
    
    if neg_word_count > 0:
        sentiment_penalty = min(0.1 * neg_word_count, 0.5)  # Cap at 0.5
        enhanced_sentiment['compound'] = max(enhanced_sentiment['compound'] - sentiment_penalty, -1.0)
        enhanced_sentiment['neg'] += sentiment_penalty
    
    # Apply voice feature adjustments if available
    if avg_pitch is not None and pitch_variance is not None:
        # Simple voice-based adjustments (if we have voice features)
        if avg_pitch > 180:  # Higher pitch
            enhanced_sentiment['compound'] = min(enhanced_sentiment['compound'] + 0.1, 1.0)
        elif avg_pitch < 120:  # Lower pitch
            enhanced_sentiment['compound'] = max(enhanced_sentiment['compound'] - 0.1, -1.0)
        
        if pitch_variance > 50:  # High variance = more emotional
            if enhanced_sentiment['compound'] > 0:
                enhanced_sentiment['compound'] *= 1.2  # Amplify positive
            elif enhanced_sentiment['compound'] < 0:
                enhanced_sentiment['compound'] *= 1.2  # Amplify negative
    
    # Normalize the pos/neg/neu values to sum to 1.0
    total = enhanced_sentiment['pos'] + enhanced_sentiment['neg'] + enhanced_sentiment['neu']
    if total > 0:
        scale_factor = 1.0 / total
        enhanced_sentiment['pos'] *= scale_factor
        enhanced_sentiment['neg'] *= scale_factor
        enhanced_sentiment['neu'] *= scale_factor
    
    return enhanced_sentiment

# Fix 9: Completely rewritten audio recording handler with robust error handling
def process_audio_recording(audio_data):
    """
    Process recorded audio data with comprehensive error handling and validation
    """
    if DEBUG:
        if audio_data:
            print(f"Received audio data: {type(audio_data)} with keys: {audio_data.keys() if isinstance(audio_data, dict) else 'Not a dict'}")
            print(f"Audio data size: {len(audio_data.get('bytes', b'')) if isinstance(audio_data, dict) and 'bytes' in audio_data else 'Unknown'} bytes")
        else:
            print("No audio data received")
    
    # Fix 10: More detailed validation of audio data
    if not audio_data:
        st.warning("No recording detected. Please click 'Start Recording' and speak.")
        return None, None, None, None
    
    # Handle both possible formats from streamlit-mic-recorder
    audio_bytes = None
    if isinstance(audio_data, dict):
        if 'bytes' in audio_data:
            audio_bytes = audio_data['bytes']
        elif 'raw' in audio_data:
            audio_bytes = audio_data['raw']
    elif isinstance(audio_data, bytes):
        audio_bytes = audio_data
    
    if not audio_bytes:
        st.warning("Recording format not recognized. Please try again in Chrome or Edge browser.")
        return None, None, None, None
    
    # Fix 11: More lenient size check but still ensure we have data
    if len(audio_bytes) < 50:  # Very small file
        st.warning("Recording too short or empty. Please record for at least 1-2 seconds.")
        return None, None, None, None
    
    with st.spinner("Processing your recording..."):
        status_msg = st.empty()
        status_msg.info("Converting and analyzing audio...")
        
        audio_path = None
        text, sentiment, avg_pitch, pitch_variance = None, None, None, None
        
        try:
            # Write audio data to a temporary file with the correct extension
            # Try to determine format from audio_data if available
            suffix = ".webm"  # Default format
            if isinstance(audio_data, dict) and 'format' in audio_data:
                suffix = f".{audio_data['format']}"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                audio_path = temp_file.name
                temp_file.write(audio_bytes)
                temp_file.flush()
            
            if DEBUG:
                print(f"Temporary audio file created: {audio_path} with size {os.path.getsize(audio_path)} bytes")
            
            # Verify the file exists and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 50:
                if DEBUG:
                    print(f"File either doesn't exist or is too small: {os.path.exists(audio_path)}, Size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'}")
                st.warning("Something went wrong with the recording. Please try again.")
                return None, None, None, None
            
            # Display audio player with error handling
            try:
                st.audio(audio_bytes)
            except Exception as audio_error:
                if DEBUG:
                    print(f"Error displaying audio: {audio_error}")
                # Continue anyway - this isn't critical
            
            # Run our improved analysis
            try:
                text, sentiment, avg_pitch, pitch_variance = analyze_speech(audio_path)
                
                # Apply enhanced sentiment analysis
                if text and text != "No speech detected":
                    sentiment = get_enhanced_sentiment(text, avg_pitch, pitch_variance)
                    
                    # Extra validation on sentiment output
                    if not isinstance(sentiment, dict) or 'compound' not in sentiment:
                        sentiment = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
                        if DEBUG:
                            print("Sentiment analysis returned invalid format, using default")
            except Exception as analysis_error:
                if DEBUG:
                    print(f"Error in speech analysis: {analysis_error}")
                text = "Error analyzing speech. Please try again."
                sentiment = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            if DEBUG:
                print(f"Error in process_audio_recording: {str(e)}")
                print(error_details)
            st.error(f"Error processing audio: {str(e)}")
            
            # Still return something valid to prevent UI errors
            text = "Error processing audio"
            sentiment = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
            avg_pitch = 150
            pitch_variance = 30
            
        finally:
            status_msg.empty()
            # Clean up temp file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception as e:
                    if DEBUG:
                        print(f"Failed to delete temp file: {e}")
    
    return text, sentiment, avg_pitch, pitch_variance