import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

# Set page configuration
st.set_page_config(
    page_title="🎥 Real-Time Attendance System",
    page_icon="📋",
    layout="wide",
)

# Page header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Real-Time Attendance System</h1>", unsafe_allow_html=True)

# Redis DB retrieval spinner
with st.spinner("Retrieving data from Redis DB..."):
    redis_face_db = face_rec.retrive_data(name="academy:register")
    st.success("✅ Data successfully retrieved from Redis!")
    st.dataframe(redis_face_db)

# Set timer for saving logs
waitTime = 30  # Time in seconds
setTime = time.time()
realtimepred = face_rec.RealTimePred()  # Real-time prediction class

# Real-Time Prediction
st.markdown("### 📸 Start Real-Time Predictions")
st.info("Ensure your camera is connected. The system updates every 30 seconds.")

# Callback function for video processing
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24")  # Convert to 3D numpy array
    
    pred_img = realtimepred.face_prediction(
        img,
        redis_face_db,
        "facial_features",
        ["Name", "Role"],
        thresh=0.5,
    )

    # Save logs every `waitTime` seconds
    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time()
        st.balloons()  # Celebrate data being saved
        st.success("✅ Logs successfully saved to Redis DB!")
    
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# Streamlit WebRTC for live video
webrtc_streamer(
    key="realtimePrediction",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
