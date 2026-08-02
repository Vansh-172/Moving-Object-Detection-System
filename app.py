import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# Load YOLO26 model
model = YOLO("yolo26x.pt")

class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Run tracking on the current frame
        results = model.track(source=img, persist=True, tracker="bytetrack.yaml")

        # Return annotated frame back to the browser UI
        return results[0].plot()

st.title("Live Camera Object Detection & Tracking (YOLO26)")
webrtc_streamer(key="yolo-stream", video_transformer_factory=YOLOVideoTransformer)