import cv2
import mediapipe as mp
import json
import os

def extract_landmarks(video_path, jsonl_path, sample_delay=0.15):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_step = max(1, round(fps * sample_delay))

    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True) as holistic:

        frame_idx = 0
        saved_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            if frame_idx % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                def get_coords(landmarks):
                    if not landmarks: return None
                    # Rounding to 4 decimals for efficiency
                    return [{"x": round(l.x, 4), "y": round(l.y, 4), "z": round(l.z, 4)} for l in landmarks.landmark]

                frame_data = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "face": get_coords(results.face_landmarks),
                    "pose": get_coords(results.pose_landmarks), # Added Pose (includes elbows)
                    "hand1": get_coords(results.right_hand_landmarks),
                    "hand2": get_coords(results.left_hand_landmarks)
                }

                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(frame_data) + '\n')
                saved_count += 1

            frame_idx += 1

    cap.release()
    print(f"✅ Extraction complete. Saved {saved_count} frames to {jsonl_path}")

if __name__ == "__main__":
    extract_landmarks("../DATASET/test/0a2ffece-2832-4011-b656-915f39aa7850.mp4", "output_landmarks.jsonl")