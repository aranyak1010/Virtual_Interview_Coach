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
from speech_analysis import process_audio_recording, get_enhanced_sentiment
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

    col1, col2 = st.columns([3, 3])

    with col1:
        st.markdown("### Microphone Check")
    
    # Use the streamlit-mic-recorder with minimal settings
        audio_data = mic_recorder(
            key="speech-recorder",
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            format="webm",  # Use webm format which is more reliable
            use_container_width=True
        )
    
    # If we have audio data, process it with our imported function
        if audio_data:
        # Here's where we call the imported function
            text, sentiment, avg_pitch, pitch_variance = process_audio_recording(audio_data)
        
            if text and text != "No speech detected":
                st.write("**📝 Transcription:**", text)
            
                sentiment_score = sentiment["compound"]
                if sentiment_score > 0.1:
                    sentiment_label, sentiment_color = "Positive 😊", "green"
                elif sentiment_score < -0.1:
                    sentiment_label, sentiment_color = "Negative 😔", "red"
                else:
                    sentiment_label, sentiment_color = "Neutral 😐", "gray"
                
                st.markdown(
                    f"**📈 Sentiment:** {sentiment_label} (Score: {sentiment_score:.2f})",
                    unsafe_allow_html=True)
                st.markdown(
                    f"**Sentiment Breakdown:** Positive: {sentiment['pos']:.2f}, Negative: {sentiment['neg']:.2f}, Neutral: {sentiment['neu']:.2f}")
                
                sentiment_meter = st.progress(0)
                normalized_sentiment = (sentiment_score + 1) / 2
                sentiment_meter.progress(normalized_sentiment)
                
                pitch_category = "High" if avg_pitch > 180 else "Medium" if avg_pitch > 120 else "Low"
                st.write(f"**🔊 Average Pitch:** {avg_pitch:.2f} Hz ({pitch_category})")
                
                variation_category = "Monotonous" if pitch_variance < 10 else "Normal" if pitch_variance < 50 else "Expressive"
                st.write(f"**📊 Voice Variation:** {pitch_variance:.2f} (Type: {variation_category})")
            elif text == "No speech detected":
                st.warning("No speech was detected in your recording. Please try again and speak clearly.")
    
    with col2:
        st.info("**Tips for better speech recording:**\n\n"
                "• Click **Start Recording** to begin\n\n"
                "• Speak clearly and confidently\n\n"
                "• Click **Stop Recording** to finish\n\n"
                "• The recording will be automatically analyzed\n\n"
                "• Keep background noise to a minimum for best results.")
                
        # Add troubleshooting tips
        with st.expander("Troubleshooting Audio Issues"):
            st.markdown("""
            **If your recording isn't being processed:**
                
            1. **Try a shorter recording** - Just 5-10 seconds is enough
            2. **Make sure your microphone is working** - Test it in another app
            3. **Check browser permissions** - Make sure your browser has microphone access
            4. **Speak louder** - The app needs to detect speech to process it
            5. **Try a different browser** - Chrome works best with audio recording
            6. **Restart the app** - Sometimes a fresh start helps
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