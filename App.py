import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from deepface import DeepFace
import librosa
import numpy as np
import tempfile
import cv2
import av
import sounddevice as sd
import wavio
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch.nn.functional as F  # For softmax

# UI Customization
st.set_page_config(page_title="Multimodal Emotion Detector", layout="wide")

st.markdown(
    """
    <h1 style="text-align: center; color: #4A90E2;"> Multimodal Emotion Detection</h1>
    <p style="text-align: center; font-size: 18px;">Upload or record text, audio, image, or video to analyze emotions.</p>
    """,
    unsafe_allow_html=True
)

# Model Paths (placeholders)
face_model_path = "face.pth"
audio_model_path = "audio.keras"
text_model_path = "text.h5"

with st.expander(" Loaded Models"):
    st.write(f"*Face Model:* {face_model_path}")
    st.write(f"*Audio Model:* {audio_model_path}")
    st.write(f"*Text Model:* {text_model_path}")

# ==========================================
# Load TEXT EMOTION MODEL
# ==========================================
emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        labels = list(emotion_model.config.id2label.values())
        top_idx = int(np.argmax(probs))
        return labels[top_idx], probs[top_idx], dict(zip(labels, probs))

# ==========================================
# Load AUDIO EMOTION MODEL
# ==========================================
audio_model_name = "superb/wav2vec2-large-superb-er"
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_name)
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)

# Layout tabs including fusion as last tab
tabs = st.tabs([" Text", " Audio", " Image", " Video", " Fusion"])

# ==========================================
# TEXT EMOTION ANALYSIS TAB
# ==========================================
with tabs[0]:
    st.subheader(" Text Emotion Analysis")
    text_input = st.text_area("Enter text for emotion detection:", height=150)

    if st.button("Analyze Text", key="text_button"):
        if text_input:
            with st.spinner("Analyzing text..."):
                emotion, confidence, probs = predict_emotion(text_input)
                st.success(" Text analysis complete!")
                st.write(f"*Detected Emotion:* {emotion} with {confidence:.2f}% confidence")
                # Save probabilities for fusion
                st.session_state['text_probs'] = probs
        else:
            st.warning(" Please enter some text.")

# ==========================================
# AUDIO EMOTION ANALYSIS TAB
# ==========================================
with tabs[1]:
    st.subheader(" Audio Emotion Analysis")

    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    def record_audio(filename="recorded.wav", duration=5, fs=22050):
        with st.spinner("Recording... Please speak!"):
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
            sd.wait()
            wavio.write(filename, recording, fs, sampwidth=2)
        st.success(" Recording complete!")
        return filename

    if st.button("Record and Analyze"):
        audio_path = record_audio()
    else:
        audio_path = None

    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_audio.read())
            audio_path = temp_audio.name

    if audio_path:
        with st.spinner("Processing audio..."):
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = audio_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = audio_model(**inputs).logits
        
        probs_tensor = F.softmax(logits, dim=1).squeeze()
        probs = probs_tensor.cpu().numpy()
        emotions = ["neutral", "happy", "sad", "angry"]
        audio_emotion_probs = {emotions[i]: float(probs[i]) for i in range(len(emotions))}
        st.success(" Audio analysis complete!")
        st.write(f"*Detected Emotions:*", audio_emotion_probs)
        # Save probs for fusion
        st.session_state['audio_probs'] = audio_emotion_probs

# ==========================================
# IMAGE EMOTION ANALYSIS TAB
# ==========================================
with tabs[2]:
    st.subheader(" Image Emotion Analysis")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

    with col2:
        captured_image = st.camera_input("Take a photo")

    image_path = None
    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            temp_img.write(uploaded_image.read())
            image_path = temp_img.name
    elif captured_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            temp_img.write(captured_image.read())
            image_path = temp_img.name

    if image_path:
        with st.spinner("Analyzing image..."):
            image_emotion = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
            dominant_emotion = image_emotion[0]['dominant_emotion']
            emotion_scores = image_emotion[0]['emotion']
            st.success(" Image analysis complete!")
            st.write(f"*Detected Emotion:*", dominant_emotion)
            # Save probs for fusion (normalize to sum 1)
            total = sum(emotion_scores.values())
            normalized_scores = {k: v/total for k, v in emotion_scores.items()}
            st.session_state['image_probs'] = normalized_scores

# ==========================================
# VIDEO EMOTION ANALYSIS TAB
# ==========================================
with tabs[3]:
    st.subheader(" Video Emotion Analysis")

    class VideoEmotionTransformer(VideoTransformerBase):
        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            if result:
                emotion = result[0]['dominant_emotion']
                cv2.putText(img, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            return img

    webrtc_streamer(key="video", video_transformer_factory=VideoEmotionTransformer)

    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with st.spinner("Processing video..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_video.read())
                temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        frame_emotions = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if result:
                frame_emotions.append(result[0]['dominant_emotion'])
            progress_bar.progress(i / total_frames)

        cap.release()
        progress_bar.empty()

        if frame_emotions:
            most_common_emotion = max(set(frame_emotions), key=frame_emotions.count)
            st.success(" Video analysis complete!")
            st.write(f"*Detected Emotion:*", most_common_emotion)
            
            # Create simple prob dict for fusion: dominant emotion 1.0, others 0
            emotions_list = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            video_probs = {emo: (1.0 if emo == most_common_emotion else 0.0) for emo in emotions_list}
            st.session_state['video_probs'] = video_probs
        else:
            st.warning(" No emotions detected in the video.")

# ==========================================
# FUSION TAB - AUTOMATIC WEIGHTS & FUSION
# ==========================================
with tabs[4]:
    st.header(" Final Emotion Fusion (Auto Weights)")

    # Retrieve unimodal probabilities
    text_probs = st.session_state.get('text_probs', None)
    audio_probs = st.session_state.get('audio_probs', None)
    image_probs = st.session_state.get('image_probs', None)
    video_probs = st.session_state.get('video_probs', None)

    unimodal_probs = {
        "Text": text_probs,
        "Audio": audio_probs,
        "Image": image_probs,
        "Video": video_probs,
    }

    # Filter only those with available probs
    unimodal_probs = {k: v for k, v in unimodal_probs.items() if v is not None}

    if unimodal_probs:
        # Calculate max confidence per modality
        max_confidences = {k: max(v.values()) for k, v in unimodal_probs.items()}
        total_confidence = sum(max_confidences.values())

        # Compute weights proportional to max confidence (avoid divide by zero)
        weights = {k: (conf / total_confidence) if total_confidence > 0 else 0 for k, conf in max_confidences.items()}

        # Get union of all emotions
        all_emotions = set()
        for probs in unimodal_probs.values():
            all_emotions.update(probs.keys())
        all_emotions = sorted(all_emotions)

        # Initialize fused probabilities dict
        fused_probs = {emotion: 0.0 for emotion in all_emotions}

        # Weighted sum of probabilities
        for modality, probs in unimodal_probs.items():
            w = weights[modality]
            for emotion in all_emotions:
                fused_probs[emotion] += probs.get(emotion, 0) * w

        # Final predicted emotion & confidence
        final_emotion = max(fused_probs, key=fused_probs.get)
        final_confidence = fused_probs[final_emotion]

        st.write("### Auto-calculated weights (based on max confidence):")
        for modality, w in weights.items():
            st.write(f"- **{modality}**: {w:.2f}")

        st.write("### Fused Emotion Probabilities:")
        st.bar_chart(fused_probs)

        st.success(f"📈 **Final Predicted Emotion:** {final_emotion} ({final_confidence:.2f}% confidence)")

    else:
        st.info("Please perform emotion analysis on at least one modality (Text, Audio, Image, or Video) to enable fusion.")