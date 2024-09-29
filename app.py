from flask import Flask, request, jsonify
import os
import cv2
from google.cloud import storage
import mediapipe as mp
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    SafetySetting,
    HarmBlockThreshold,
    HarmCategory
)
from math import degrees, atan2

app = Flask(__name__)

# Set Google Cloud Project details
PROJECT_ID = "workoutapp-616d04"
LOCATION = "us-central1"

# Initialize MediaPipe and VertexAI
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LANDMARK_NAMES = {
    0: "NOSE",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    # Add other landmarks as necessary
}

# Configure Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.5-flash-001")
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}
safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
]

# Utility function to download a blob from GCS
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

# Extract keypoints using MediaPipe
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    if not cap.isOpened():
        print("Error: Could not open video.")
        return keypoints_list
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            keypoints = {LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z) for i, lm in enumerate(results.pose_landmarks.landmark) if i in LANDMARK_NAMES}
            keypoints_list.append((frame, keypoints))
    cap.release()
    print("Keypoint extraction completed.")
    return keypoints_list

# Calculate angle between three points
def calculate_angle(a, b, c):
    angle = degrees(atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]))
    angle = angle % 360
    if angle > 180:
        angle = 360 - angle
    return abs(angle)

# Identify the peak frame based on the workout type
def find_peak_frame(keypoints_list, workout_type):
    peak_frame = None
    peak_keypoints = None
    peak_measure = None
    for frame, keypoints in keypoints_list:
        if workout_type in ["squat", "bulgarian_split_squat"]:
            hip_y = max(keypoints['LEFT_HIP'][1], keypoints['RIGHT_HIP'][1])
            if peak_measure is None or hip_y > peak_measure:
                peak_measure = hip_y
                peak_frame = frame
                peak_keypoints = keypoints
        elif workout_type in ["rdl", "deadlift", "hip_thrust"]:
            hip_y = min(keypoints['LEFT_HIP'][1], keypoints['RIGHT_HIP'][1])
            if peak_measure is None or hip_y < peak_measure:
                peak_measure = hip_y
                peak_frame = frame
                peak_keypoints = keypoints
        elif workout_type in ["bench_press", "rows"]:
            shoulder_y = max(keypoints['LEFT_SHOULDER'][1], keypoints['RIGHT_SHOULDER'][1])
            if peak_measure is None or shoulder_y > peak_measure:
                peak_measure = shoulder_y
                peak_frame = frame
                peak_keypoints = keypoints
    return peak_frame, peak_keypoints

# Analyze keypoints with Gemini at the peak position
def analyze_with_gemini(peak_frame, peak_keypoints, workout_type):
    hip_angle = calculate_angle(peak_keypoints['LEFT_SHOULDER'], peak_keypoints['LEFT_HIP'], peak_keypoints['LEFT_KNEE'])
    knee_angle = calculate_angle(peak_keypoints['LEFT_HIP'], peak_keypoints['LEFT_KNEE'], peak_keypoints['LEFT_ANKLE'])
    shoulder_angle = calculate_angle(peak_keypoints['RIGHT_SHOULDER'], peak_keypoints['LEFT_SHOULDER'], peak_keypoints['LEFT_HIP'])
    prompt = (
        f"The user is performing a {workout_type} exercise. The angles measured at the peak position are as follows:\n"
        f"- Hip angle: {hip_angle:.2f} degrees\n"
        f"- Knee angle: {knee_angle:.2f} degrees\n"
        f"- Shoulder angle: {shoulder_angle:.2f} degrees\n"
        "This data was recorded during the peak phase of the movement. The user is an intermediate lifter. "
        "Based on this information, please analyze the form and provide CONCISE feedback on whether "
        "these angles are appropriate and how the user can improve their form."
    )
    response = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )
    feedback = response.text if response.text else "No feedback received."
    return feedback, peak_frame, peak_keypoints

# Save the frame with incorrect form, display it, and upload to GCS
def save_incorrect_frame(frame, keypoints, save_path, bucket_name, destination_blob_name):
    for keypoint in keypoints.values():
        x, y, _ = keypoint
        cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, frame)
    print(f"Saved incorrect form frame to {save_path}.")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(save_path)
    print(f"Uploaded {save_path} to {destination_blob_name} in {bucket_name} bucket.")

# Main API route
@app.route('/analyze', methods=['POST'])
def analyze_video():
    video_file = request.files['video']
    workout_type = request.form['workout_type']
    bucket_name = "video-bucket333"
    
    video_path = "uploaded_video.mp4"
    video_file.save(video_path)

    keypoints_list = extract_keypoints(video_path)
    if not keypoints_list:
        return jsonify({"error": "No keypoints were extracted. Check the video path or content."}), 400

    peak_frame, peak_keypoints = find_peak_frame(keypoints_list, workout_type)
    if peak_frame is None or peak_keypoints is None:
        return jsonify({"error": "No peak frame detected. Unable to analyze form."}), 400

    feedback, incorrect_frame, incorrect_keypoints = analyze_with_gemini(peak_frame, peak_keypoints, workout_type)

    save_path = "incorrect_frame.png"
    save_incorrect_frame(incorrect_frame, incorrect_keypoints, save_path, bucket_name, "incorrect_frame.png")

    return jsonify({"feedback": feedback})

# Run the Flask app
if __name__ == "__main__":
   port = int(os.environ.get("PORT", 8080))
   app.run(host="0.0.0.0", port=port)

    