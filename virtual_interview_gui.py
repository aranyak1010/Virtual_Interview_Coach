import streamlit as st
import cv2
import mediapipe as mp
import speech_recognition as sr
import librosa
import numpy as np
import soundfile as sf
import tempfile
import nltk
import pytesseract
from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.io.wavfile import write
from streamlit_mic_recorder import mic_recorder
import tempfile
import time
import threading
import os

nltk.download('vader_lexicon')

# Initialize Mediapipe and Sentiment Analyzer
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
sia = SentimentIntensityAnalyzer()
bert_model = SentenceTransformer("all-MiniLM-L6-v2")


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



# Function to match resume with job description (No changes needed here)
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

# Modified speech analysis using browser-based recording
def speech_analysis_ui():
    st.header("🎙️ Speech Analysis")
    init_recording_state()

    # Browser-based audio recording
    audio_data = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        format="webm"
    )

    if audio_data:
        # Convert webm to wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            write(tf.name, audio_data['sample_rate'], audio_data['bytes'])
            st.session_state.temp_audio_path = tf.name

        # Playback and analysis
        st.audio(audio_data['bytes'], format="audio/webm")
        analyze_audio()

# Modified analysis function
def analyze_audio():
    if st.session_state.temp_audio_path:
        try:
            text, sentiment, avg_pitch, pitch_variance = analyze_speech(st.session_state.temp_audio_path)
            
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
                
                # Visual feedback
                pitch_status = "✅ Good Variation" if pitch_variance > 20 else "⚠️ Monotonous"
                sentiment_status = "😊 Positive" if sentiment['compound'] > 0.05 else "😐 Neutral" if sentiment['compound'] > -0.05 else "😟 Negative"
                
                st.metric("Sentiment", sentiment_status)
                st.metric("Pitch Variation", pitch_status)
            else:
                st.warning("No speech detected. Please try again.")
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Modified analyze_speech function
def analyze_speech(audio_path):
    recognizer = sr.Recognizer()
    text = "No speech recognized"
    sentiment = {"compound": 0}
    avg_pitch = 0.0
    pitch_variance = 0.0

    try:
        # Speech to text
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        
        # Pitch analysis
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) > 0:
            f0, voiced_flag, _ = librosa.pyin(y, 
                                            fmin=librosa.note_to_hz('C2'),
                                            fmax=librosa.note_to_hz('C7'))
            voiced_pitches = f0[voiced_flag]
            avg_pitch = np.mean(voiced_pitches) if len(voiced_pitches) > 0 else 0
            pitch_variance = np.var(voiced_pitches) if len(voiced_pitches) > 0 else 0
        
        # Sentiment analysis
        sentiment = get_enhanced_sentiment(text, avg_pitch, pitch_variance)

    except sr.UnknownValueError:
        st.warning("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"Speech service error: {e}")
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

    return text, sentiment, avg_pitch, pitch_variance

# Enhanced sentiment analysis function with more sensitivity
def get_enhanced_sentiment(text, avg_pitch=None, pitch_variance=None):
    """
    Calculate sentiment with enhanced sensitivity and voice tone adjustment
    """
    # Get the base sentiment from VADER
    base_sentiment = sia.polarity_scores(text)
    
    # Apply a sensitivity multiplier to make sentiment more pronounced
    # This helps avoid the "neutral" trap for short or ambiguous phrases
    sensitivity = 1.5  # Amplify the sentiment
    
    # Apply to compound score (but keep within -1 to 1 range)
    enhanced_sentiment = base_sentiment.copy()
    enhanced_sentiment["compound"] = max(min(base_sentiment["compound"] * sensitivity, 1.0), -1.0)
    
    # Also adjust the pos/neg values while maintaining their sum with neu
    total = base_sentiment["pos"] + base_sentiment["neg"] + base_sentiment["neu"]
    
    if base_sentiment["compound"] > 0:
        # For positive sentiment, boost pos score
        enhanced_sentiment["pos"] = min(base_sentiment["pos"] * sensitivity, total)
        enhanced_sentiment["neu"] = max(total - enhanced_sentiment["pos"] - enhanced_sentiment["neg"], 0)
    elif base_sentiment["compound"] < 0:
        # For negative sentiment, boost neg score
        enhanced_sentiment["neg"] = min(base_sentiment["neg"] * sensitivity, total)
        enhanced_sentiment["neu"] = max(total - enhanced_sentiment["pos"] - enhanced_sentiment["neg"], 0)
    
    # Adjust sentiment based on voice pitch and variance when available
    if avg_pitch is not None:
        # Deep voice adjustment: Deep voices can be misinterpreted as positive
        # when they're often neutral or negative
        if avg_pitch < 120:  # Deep voice threshold
            # Reduce positive bias for deep voices
            pitch_adjustment = -0.15  # Slight negative adjustment for deep voices
            enhanced_sentiment["compound"] = max(min(enhanced_sentiment["compound"] + pitch_adjustment, 1.0), -1.0)
            
            # Increase neutral component for deep voices
            if enhanced_sentiment["compound"] > -0.2 and enhanced_sentiment["compound"] < 0.2:
                enhanced_sentiment["neu"] = min(enhanced_sentiment["neu"] + 0.2, 1.0)
                # Adjust pos/neg to maintain proportions
                total_adjusted = enhanced_sentiment["pos"] + enhanced_sentiment["neg"] + enhanced_sentiment["neu"]
                scale_factor = 1.0 / total_adjusted if total_adjusted > 0 else 1.0
                enhanced_sentiment["pos"] *= scale_factor
                enhanced_sentiment["neg"] *= scale_factor
                enhanced_sentiment["neu"] *= scale_factor
    
    # Adjust for monotonous speech when pitch variance is available
    if pitch_variance is not None and pitch_variance < 10:  # Low variance = monotonous
        # Monotonous speech tends toward neutral regardless of words
        neutrality_boost = 0.25  # Boost neutral component for monotonous speech
        
        # Shift compound score toward neutral (0)
        enhanced_sentiment["compound"] *= (1 - neutrality_boost)
        
        # Increase neu component
        enhanced_sentiment["neu"] = min(enhanced_sentiment["neu"] + neutrality_boost, 1.0)
        
        # Rescale to ensure sum = 1
        total_adjusted = enhanced_sentiment["pos"] + enhanced_sentiment["neg"] + enhanced_sentiment["neu"]
        if total_adjusted > 0:
            scale_factor = 1.0 / total_adjusted
            enhanced_sentiment["pos"] *= scale_factor
            enhanced_sentiment["neg"] *= scale_factor
            enhanced_sentiment["neu"] *= scale_factor
    
    return enhanced_sentiment

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
    
    # Speech Analysis Section
    st.header("🎙️ Speech Analysis")
    
    # Use a wider column ratio to give more space for the tips
    col1, col2 = st.columns([3, 3])
    
    with col1:
        st.markdown("### Test Your Audio First")
        st.markdown("Check if your microphone is working properly:")
        
        if st.button("Test Microphone", key="test_mic"):
            try:
                with sr.Microphone() as source:
                    st.success("Microphone connected and working!")
            except Exception as e:
                st.error(f"Microphone error: {str(e)}")
        
        st.markdown("---")
        
        # Initialize recording state
        init_recording_state()
        
        # Create a placeholder for status messages
        status_placeholder = st.empty()
        
        # NEW: Recording control buttons
        col_rec, col_stop, col_process = st.columns(3)
        
        with col_rec:
            # Play button - starts recording
            if st.button("▶️ Start Recording", disabled=st.session_state.is_recording):
                start_recording(status_placeholder)
        
        with col_stop:
            # Stop button - stops recording
            if st.button("⏹️ Stop Recording", disabled=not st.session_state.is_recording):
                stop_recording()
                status_placeholder.info("Recording stopped. Click 'Process Recording' to analyze.")
        
        with col_process:
            # Process button - processes recorded audio
            if st.button("🔄 Process Recording", disabled=st.session_state.is_recording):
                with st.spinner("Processing your recording..."):
                    audio_path = process_audio_chunks()
                    
                    if audio_path:
                        status_placeholder.success("Audio processed successfully!")
                        
                        # Display the recorded audio
                        st.subheader("Your Recording:")
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format="audio/wav")
                        
                        # Analyze the speech
                        text, sentiment, avg_pitch, pitch_variance = analyze_speech(audio_path)
                        
                        if text and text != "No speech recognized":
                            st.write("**📝 Transcription:**", text)
                            
                            # Display sentiment with enhanced thresholds for classification
                            sentiment_score = sentiment["compound"]
                            
                            # Use tighter thresholds to avoid too many "neutral" results
                            # But also account for deep voice and monotonous speech
                            if avg_pitch < 120 or pitch_variance < 10:
                                # For deep voices or monotonous speech, use wider neutral range
                                if sentiment_score > 0.2:  # Higher threshold for positive
                                    sentiment_label = "Positive 😊"
                                    sentiment_color = "green"
                                elif sentiment_score < -0.15:  # Higher threshold for negative
                                    sentiment_label = "Negative 😔"
                                    sentiment_color = "red"
                                else:
                                    sentiment_label = "Neutral 😐"
                                    sentiment_color = "gray"
                            else:
                                # Regular thresholds for normal speech
                                if sentiment_score > 0.1:
                                    sentiment_label = "Positive 😊"
                                    sentiment_color = "green"
                                elif sentiment_score < -0.1:
                                    sentiment_label = "Negative 😔"
                                    sentiment_color = "red"
                                else:
                                    sentiment_label = "Neutral 😐"
                                    sentiment_color = "gray"
                            
                            # Display with more detail
                            st.markdown(f"**📈 Sentiment:** <span style='color:{sentiment_color}'>{sentiment_label}</span> (Score: {sentiment_score:.2f})", unsafe_allow_html=True)
                            
                            # Show the components for more insight
                            st.markdown(f"**Sentiment Breakdown:** Positive: {sentiment['pos']:.2f}, Negative: {sentiment['neg']:.2f}, Neutral: {sentiment['neu']:.2f}")
                            
                            # Create a sentiment meter
                            sentiment_meter = st.progress(0)
                            normalized_sentiment = (sentiment_score + 1) / 2  # Convert from [-1,1] to [0,1]
                            sentiment_meter.progress(normalized_sentiment)
                            
                            if avg_pitch > 0:
                                # Categorize pitch
                                pitch_category = "High" if avg_pitch > 180 else "Medium" if avg_pitch > 120 else "Low"
                                st.write(f"**🔊 Average Pitch:** {avg_pitch:.2f} Hz ({pitch_category})")
                            
                            # Add pitch variance display
                            if pitch_variance > 0:
                                variation_category = "Monotonous" if pitch_variance < 10 else "Normal" if pitch_variance < 50 else "Expressive"
                                st.write(f"**📊 Voice Variation:** {pitch_variance:.2f} (Type: {variation_category})")
                        else:
                            st.warning("No speech was recognized. Please try again and speak clearly into the microphone.")
                            
                            # Provide troubleshooting advice
                            st.info("Listen to your recording above to check audio quality. If you hear your voice clearly but it's not being recognized, try adjusting your microphone settings.")
                    else:
                        status_placeholder.error("No audio recorded. Try recording again.")
        
        # Show recording status
        if st.session_state.is_recording:
            # Display a visual indicator that recording is in progress
            st.markdown("#### 🔴 Recording in progress...")
            # Show recording duration
            if 'start_time' not in st.session_state:
                st.session_state.start_time = time.time()
            
            elapsed = time.time() - st.session_state.start_time
            st.text(f"Recording duration: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
    
    with col2:
        st.info("Tips for better speech recording:\n\n"
                "• Click ▶️ Start Recording to begin\n\n"
                "• Speak clearly at a moderate pace\n\n"
                "• Click ⏹️ Stop Recording when finished\n\n"
                "• Click 🔄 Process Recording to analyze\n\n"
                "• Reduce background noise for best results")

if __name__ == "__main__":
    main()