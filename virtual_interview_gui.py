import streamlit as st
import cv2
import mediapipe as mp
import speech_recognition as sr
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import tempfile
import nltk
import pytesseract
import subprocess
from streamlit_mic_recorder import mic_recorder
from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.io import wavfile
from scipy.io.wavfile import write
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
from pydub import AudioSegment
from speech_analysis import analyze_speech_simplified, process_audio_recording, get_enhanced_sentiment
import pyaudio
import tempfile
import time
import threading
import os
import io
import time

nltk.download('vader_lexicon')

# Initialize Mediapipe and Sentiment Analyzer
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
sia = SentimentIntensityAnalyzer()
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Setup Variables ----
audio_chunks = []

# ---- Audio Processor to Collect Audio ----
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recording = False

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        global audio_chunks

        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0  # normalize 16-bit PCM
        if self.recording:
            audio_chunks.append(audio)
        return frame


# Function to analyze facial expressions and engagement
def analyze_face():
    st.write("Enable camera and look at the screen for analysis.")
    img_file = st.camera_input("Capture Image")

    if img_file:
        # Read image and convert to RGB
        image = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 1)
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                st.write("✅ Face Detected! Analyzing expressions...")

                face_landmarks = results.multi_face_landmarks[0].landmark
                
                # ---------- IMPROVED EYE CLOSURE DETECTION FOR GLASSES ----------
                # Using multiple points around the eye to improve glasses detection
                
                # Left eye landmarks (focusing on outer and inner corners + lids)
                left_eye_inner = face_landmarks[33]  # Inner corner
                left_eye_outer = face_landmarks[133]  # Outer corner
                left_eye_top1 = face_landmarks[159]  # Upper eyelid center
                left_eye_top2 = face_landmarks[160]  # Upper eyelid additional point
                left_eye_bottom1 = face_landmarks[145]  # Lower eyelid center
                left_eye_bottom2 = face_landmarks[144]  # Lower eyelid additional point
                
                # Right eye landmarks
                right_eye_inner = face_landmarks[362]  # Inner corner
                right_eye_outer = face_landmarks[263]  # Outer corner
                right_eye_top1 = face_landmarks[386]  # Upper eyelid center
                right_eye_top2 = face_landmarks[385]  # Upper eyelid additional point
                right_eye_bottom1 = face_landmarks[374]  # Lower eyelid center
                right_eye_bottom2 = face_landmarks[373]  # Lower eyelid additional point
                
                # Mouth landmarks
                upper_lip = face_landmarks[13]  # Upper lip
                lower_lip = face_landmarks[14]  # Lower lip
                left_mouth = face_landmarks[61]  # Left corner of mouth
                right_mouth = face_landmarks[291]  # Right corner of mouth
                
                # Face reference points
                nose_tip = face_landmarks[1]  # Tip of nose
                chin = face_landmarks[152]  # Bottom of chin
                left_cheek = face_landmarks[50]  # Left cheek
                right_cheek = face_landmarks[280]  # Right cheek
                forehead = face_landmarks[10]  # Forehead
                
                # Calculate face dimensions for normalization
                face_height = abs(forehead.y - chin.y)
                face_width = abs(left_cheek.x - right_cheek.x)
                
                # ---------- IMPROVED EYE METRICS FOR GLASSES WEARERS ----------
                # Calculate multiple eye openness metrics and use the minimum
                
                # Left eye vertical openness (multiple measurements)
                left_eye_open1 = abs(left_eye_top1.y - left_eye_bottom1.y) / face_height
                left_eye_open2 = abs(left_eye_top2.y - left_eye_bottom2.y) / face_height
                
                # Right eye vertical openness (multiple measurements)
                right_eye_open1 = abs(right_eye_top1.y - right_eye_bottom1.y) / face_height
                right_eye_open2 = abs(right_eye_top2.y - right_eye_bottom2.y) / face_height
                
                # Take the minimum of these measurements (stricter detection)
                left_eye_opening = min(left_eye_open1, left_eye_open2)
                right_eye_opening = min(right_eye_open1, right_eye_open2)
                avg_eye_opening = (left_eye_opening + right_eye_opening) / 2
                
                # Calculate eye aspect ratio (EAR) - robust to glasses
                # EAR = (vertical distances) / (horizontal distance)
                left_eye_ear = (left_eye_open1 + left_eye_open2) / (2 * abs(left_eye_inner.x - left_eye_outer.x) / face_width)
                right_eye_ear = (right_eye_open1 + right_eye_open2) / (2 * abs(right_eye_inner.x - right_eye_outer.x) / face_width)
                avg_eye_ear = (left_eye_ear + right_eye_ear) / 2
                
                # ---------- MOUTH METRICS ----------
                # Mouth openness (vertical)
                mouth_openness = abs(upper_lip.y - lower_lip.y) / face_height
                
                # Mouth width and curvature (for smile detection)
                mouth_width = abs(left_mouth.x - right_mouth.x) / face_width
                
                # Calculate mouth corner lift (for smile detection)
                neutral_mouth_y = (upper_lip.y + lower_lip.y) / 2
                left_corner_lift = neutral_mouth_y - left_mouth.y
                right_corner_lift = neutral_mouth_y - right_mouth.y
                avg_mouth_corner_lift = (left_corner_lift + right_corner_lift) / 2 / face_height
                
                # ---------- ADAPTIVE THRESHOLDS ----------
                # These thresholds are more adaptive for glasses wearers
                EYE_CLOSED_THRESHOLD = 0.012  # When eyes are considered closed
                EYE_EAR_CLOSED_THRESHOLD = 0.15  # When EAR indicates closed eyes
                EYE_PARTIALLY_CLOSED_THRESHOLD = 0.018  # When eyes are partially closed
                MOUTH_OPEN_THRESHOLD = 0.025  # When mouth is considered open
                SMILE_THRESHOLD = 0.003  # When mouth corners are considered lifted (smiling)
                
                # ---------- IMPROVED DECISION LOGIC ----------
                # Determine eye state using multiple metrics (robust to glasses)
                eyes_closed_by_openness = avg_eye_opening < EYE_CLOSED_THRESHOLD
                eyes_closed_by_ear = avg_eye_ear < EYE_EAR_CLOSED_THRESHOLD
                
                # Use either metric to detect closed eyes (more robust)
                if eyes_closed_by_openness or eyes_closed_by_ear:
                    eye_state = "Closed"
                elif avg_eye_opening < EYE_PARTIALLY_CLOSED_THRESHOLD:
                    eye_state = "Partially Closed"
                else:
                    eye_state = "Open"
                
                # Determine mouth state
                if mouth_openness > MOUTH_OPEN_THRESHOLD:
                    mouth_state = "Open"
                else:
                    mouth_state = "Closed"
                
                # Determine smile state - more sensitive for closed mouth smiles
                is_smiling = avg_mouth_corner_lift > SMILE_THRESHOLD
                
                # ---------- ENGAGEMENT DETECTION ----------
                if eye_state == "Closed":
                    engagement_level = "Low (Drowsy/Disengaged 😴)"
                elif eye_state == "Partially Closed":
                    engagement_level = "Low to Moderate (Tired 😑)"
                elif mouth_state == "Open" and not is_smiling:
                    engagement_level = "Moderate (Talking/Responding 🗣️)"
                else:
                    engagement_level = "High (Attentive 👀)"
                
                # ---------- EXPRESSION DETECTION ----------
                if eye_state == "Closed":
                    expression = "Drowsy/Asleep 😴"
                elif is_smiling and mouth_state == "Closed":
                    expression = "Smiling (Closed Mouth) 🙂"
                elif is_smiling and mouth_state == "Open":
                    expression = "Smiling (Open Mouth) 😀"
                elif mouth_state == "Open":
                    expression = "Talking/Surprised 😮"
                else:
                    expression = "Neutral 😐"
                
                
                # ---------- FINAL OUTPUT ----------
                st.write(f"👀 **Engagement Level:** {engagement_level}")
                st.write(f"🙂 **Facial Expression:** {expression}")
                
            else:
                st.write("❌ No face detected. Try again.")


def extract_text_from_pdf(pdf_path):
    try:
        # Remove hardcoded poppler_path - Streamlit Cloud will handle via packages.txt
        images = convert_from_path(pdf_path, dpi=500)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from resume: {e}")
        return ""


# Function to match resume with job description
def job_description_matching(resume_text, job_description):
    try:
        resume_embedding = bert_model.encode(resume_text, convert_to_numpy=True)
        job_embedding = bert_model.encode(job_description, convert_to_numpy=True)
        similarity_score = cosine_similarity(resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1))[0][0]

        if similarity_score > 0.75:
            feedback = "✅ **Great fit!** Your resume closely matches the job requirements."
        elif 0.50 < similarity_score <= 0.75:
            feedback = "⚠️ **Moderate match.** Consider adding more relevant skills or experience."
        else:
            feedback = "❌ **Weak match.** Update your resume to better align with the job description."

        return similarity_score, feedback
    except Exception as e:
        st.error(f"Job matching failed: {e}")
        return 0.0, "Error in matching."
    
# Initialize session state
def init_recording_state():
    if 'temp_audio_path' not in st.session_state:
        st.session_state.temp_audio_path = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'audio_chunks' not in st.session_state:
        st.session_state.audio_chunks = []

def start_recording(status_placeholder):
    st.session_state.is_recording = True
    st.session_state.audio_chunks = []
    st.session_state.start_time = time.time()
    status_placeholder.info("🔴 Recording... Speak clearly into your microphone.")
    
    # Start the recording thread
    threading.Thread(target=record_audio).start()

def record_audio():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            
            while st.session_state.is_recording:
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    # Save to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio.get_wav_data())
                        st.session_state.audio_chunks.append(f.name)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Recording error: {e}")
                    break
    except Exception as e:
        print(f"Microphone error: {e}")
        st.session_state.is_recording = False

def stop_recording():
    st.session_state.is_recording = False
    # Allow a small delay for the recording thread to finish
    time.sleep(0.5)

def process_audio_chunks():
    if not st.session_state.audio_chunks:
        return None
    
    try:
        # Create a combined audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as combined_file:
            combined_path = combined_file.name
            
        # Combine all chunks into a single audio file
        combined_audio = None
        sample_rate = None
        
        for chunk_path in st.session_state.audio_chunks:
            try:
                y, sr = librosa.load(chunk_path, sr=None)
                if combined_audio is None:
                    combined_audio = y
                    sample_rate = sr
                else:
                    combined_audio = np.concatenate((combined_audio, y))
                    
                # Clean up the temporary chunk file
                os.unlink(chunk_path)
            except Exception as e:
                print(f"Error processing chunk {chunk_path}: {e}")
        
        if combined_audio is not None and sample_rate is not None:
            sf.write(combined_path, combined_audio, sample_rate)
            st.session_state.temp_audio_path = combined_path
            return combined_path
        
        return None
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def speech_analysis_ui():
    st.header("🎙️ Speech Analysis")
    init_recording_state()

    # Browser-based audio recording
    audio_data = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        format="webm"
    )

    if audio_data and 'bytes' in audio_data and 'sample_rate' in audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            write(tf.name, audio_data['sample_rate'], np.frombuffer(audio_data['bytes'], dtype=np.int16))
            st.session_state.temp_audio_path = tf.name

        st.audio(audio_data['bytes'], format="audio/webm")
        analyze_audio()

def analyze_audio():
    if st.session_state.temp_audio_path:
        try:
            text, sentiment, avg_pitch, pitch_variance = analyze_speech_simplified(st.session_state.temp_audio_path)
            
            if text and text != "No speech recognized":
                st.subheader("📊 Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**📝 Recognized Text:**")
                    st.info(text)
                    st.write(f"**😊 Sentiment Score:** {sentiment['compound']:.2f}")
                with col2:
                    st.write("**🎵 Voice Analysis**")
                    st.write(f"• Average Pitch: {avg_pitch:.2f} Hz")
                    st.write(f"• Pitch Variance: {pitch_variance:.2f}")
                
                pitch_status = "✅ Good Variation" if pitch_variance > 20 else "⚠️ Monotonous"
                sentiment_status = "😊 Positive" if sentiment['compound'] > 0.05 else "😐 Neutral" if sentiment['compound'] > -0.05 else "😟 Negative"
                
                st.metric("Sentiment", sentiment_status)
                st.metric("Pitch Variation", pitch_status)
            else:
                st.warning("No speech detected. Please try again.")
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")


# Streamlit UI
def main():
    st.title("JobGPT - Your Personal AI-Powered Virtual Interview Coach")
    st.write("Analyze your resume, facial expressions, and speech performance.")
    
    # Resume Analysis Section
    st.header("📄 Resume Analysis")
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    job_desc = st.text_area("Paste Job Description Here")

    if uploaded_file and job_desc and st.button("Analyze Resume"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                resume_text = extract_text_from_pdf(temp_pdf.name)
                if resume_text:
                    score, feedback = job_description_matching(resume_text, job_desc)
                    st.write(f"**📊 Job Matching Score:** {score:.2f}")
                    st.write(feedback)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
    
    # Facial Analysis Section
    st.header("📷 Facial Expression Analysis")
    analyze_face()
    
    # Speech analysis section
    st.header("🎙️ Speech Analysis")

    # Fix 12: Added debug info toggle with better default
    show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
    
    col1, col2 = st.columns([3, 3])

    with col1:
        st.markdown("### Microphone Check")

        # Fix 13: Updated recorder with more compatible options and try multiple formats
        try:
            audio_data = mic_recorder(
                key="speech-recorder",
                start_prompt="🎙️ Start Recording",
                stop_prompt="⏹️ Stop Recording",
                format="webm",  # Most compatible format
                sampling_rate=16000,  # 16kHz sampling rate
                use_container_width=True,
                energy_threshold=0.01,  # More sensitive detection
                init_message="Click to start recording"
            )
        except Exception as e:
            st.error(f"Error initializing microphone recorder: {e}")
            st.info("Try refreshing the page or using a different browser.")
            audio_data = None

        # Fix 14: Add comprehensive debug information for troubleshooting
        if show_debug and audio_data:
            with st.expander("Debug Information"):
                st.write("Audio Data Type:", type(audio_data))
                if isinstance(audio_data, dict):
                    st.write("Audio Data Keys:", list(audio_data.keys()))
                    if 'bytes' in audio_data:
                        st.write(f"Audio Data Size: {len(audio_data['bytes'])} bytes")
                    if 'format' in audio_data:
                        st.write(f"Audio Format: {audio_data['format']}")
                    if 'sampling_rate' in audio_data:
                        st.write(f"Sampling Rate: {audio_data['sampling_rate']} Hz")
                else:
                    st.write("Audio Data is not a dictionary")
        
        # Process audio data if available
        if audio_data:
            text, sentiment, avg_pitch, pitch_variance = process_audio_recording(audio_data)
            
            if text:  # Checking if text is not None
                if text != "No speech detected":
                    st.write("**📝 Transcription:**", text)
                    
                    # Ensure we have valid sentiment data
                    if sentiment and isinstance(sentiment, dict) and 'compound' in sentiment:
                        sentiment_score = sentiment["compound"]
                        if sentiment_score > 0.1:
                            sentiment_label, sentiment_color = "Positive 😊", "green"
                        elif sentiment_score < -0.1:
                            sentiment_label, sentiment_color = "Negative 😔", "red"
                        else:
                            sentiment_label, sentiment_color = "Neutral 😐", "gray"
                        
                        st.markdown(
                            f"**📈 Sentiment:** <span style='color:{sentiment_color}'>{sentiment_label}</span> (Score: {sentiment_score:.2f})",
                            unsafe_allow_html=True)
                        st.markdown(
                            f"**Sentiment Breakdown:** Positive: {sentiment['pos']:.2f}, Negative: {sentiment['neg']:.2f}, Neutral: {sentiment['neu']:.2f}")
                        
                        # Fix: Use min/max to ensure value is in valid range for progress bar
                        normalized_sentiment = max(0.0, min(1.0, (sentiment_score + 1) / 2))
                        sentiment_meter = st.progress(normalized_sentiment)
                        
                        if avg_pitch is not None:
                            pitch_category = "High" if avg_pitch > 180 else "Medium" if avg_pitch > 120 else "Low"
                            st.write(f"**🔊 Average Pitch:** {avg_pitch:.2f} Hz ({pitch_category})")
                        
                        if pitch_variance is not None:
                            variation_category = "Monotonous" if pitch_variance < 10 else "Normal" if pitch_variance < 50 else "Expressive"
                            st.write(f"**📊 Voice Variation:** {pitch_variance:.2f} (Type: {variation_category})")
                elif show_debug:  
                    # For debugging, show when no speech is detected
                    st.warning("No speech was detected in your recording. Please try again and speak clearly and louder.")
                    
                    # Fix 15: Added more helpful suggestions with better formatting
                    st.info("""
                    **Tips for better speech detection:**
                    
                    • Speak louder and closer to the microphone
                    • Try using headphones with a microphone
                    • Make sure your browser has microphone permissions
                    • Try a recording of at least 2-3 seconds
                    • Test your microphone in another application first
                    """)
            elif show_debug:
                st.error("Speech analysis returned None. Check console for errors.")

    with col2:
        st.info("**Tips for better speech recording:**\n\n"
                "• Click **Start Recording** to begin\n\n"
                "• Speak clearly and directly into your microphone\n\n"
                "• Record for at least 3-5 seconds\n\n"
                "• Keep background noise to a minimum\n\n"
                "• Use Chrome or Edge for best results")
                
        # Expanded troubleshooting section
        with st.expander("Troubleshooting Audio Issues"):
            st.markdown("""
            **Common Issues and Solutions:**
            
            1. **"No speech detected" message**
               - Speak louder and closer to the microphone
               - Make sure your microphone isn't muted
               - Try a different browser (Chrome works best)
               - Check that your microphone is working in other applications
            
            2. **Browser Permission Issues**
               - Click the lock/info icon in your address bar
               - Ensure microphone permissions are set to "Allow"
               - Try clearing your browser cache and reloading
               - Restart your browser completely
            
            3. **Technical Requirements**
               - Make sure you have all required libraries installed:
                 ```
                 pip install streamlit streamlit-mic-recorder pydub nltk numpy SpeechRecognition
                 ```
               - If using Linux, you may need additional system packages:
                 ```
                 sudo apt-get install portaudio19-dev python-pyaudio python3-pyaudio
                 ```
               
            4. **Audio Format Issues**
               - Try restarting your browser
               - Some virtual environments or older browsers may have compatibility issues
               - Make sure your computer's audio drivers are up to date
               
            5. **If Nothing Works**
               - Try using an external microphone if available
               - Record at a higher volume
               - Try a different computer or device
               - Clear your browser's site data for this site
            """)
            
        # Add quick tech check button
        if st.button("🔍 Run Microphone Test"):
            st.write("Testing your microphone...")
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                
                st.success("✅ Speech recognition module loaded successfully")
                
                # Check for available microphones
                mic_list = sr.Microphone.list_microphone_names()
                st.write(f"Available Microphones: {len(mic_list)}")
                if len(mic_list) > 0:
                    st.write("First microphone detected:", mic_list[0])
                    st.success("✅ Microphone detected")
                else:
                    st.error("❌ No microphones detected")
            except Exception as e:
                st.error(f"❌ Error testing microphone: {e}")
                st.info("Try installing the SpeechRecognition package: pip install SpeechRecognition")


if __name__ == "__main__":
    main()