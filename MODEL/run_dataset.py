import pandas as pd
import cv2
import mediapipe as mp
import json
import os
from tqdm import tqdm

def extract_landmarks(video_path, sample_delay=0.15):
    """Extracts MediaPipe landmarks and returns them as a list of frames."""
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None  # Return None if the video file is missing or corrupted

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_step = max(1, round(fps * sample_delay))

    frames_list = []

    # Using model_complexity=1 for faster processing on MacBook Air
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True) as holistic:

        frame_idx = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            if frame_idx % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                def get_coords(landmarks):
                    if not landmarks: return None
                    return [{"x": round(l.x, 4), "y": round(l.y, 4), "z": round(l.z, 4)} for l in landmarks.landmark]

                frame_data = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "face": get_coords(results.face_landmarks),
                    "pose": get_coords(results.pose_landmarks),
                    "hand1": get_coords(results.right_hand_landmarks),
                    "hand2": get_coords(results.left_hand_landmarks)
                }

                frames_list.append(frame_data)

            frame_idx += 1

    cap.release()
    return frames_list


if __name__ == "__main__":
    # 1. Configuration
    csv_path = "../DATASET/annotations.csv"
    train_out = "../DATASET/final_train.jsonl"
    test_out = "../DATASET/final_test.jsonl"

    # Optional: Delete existing output files so you don't duplicate data if restarting
    if os.path.exists(train_out): os.remove(train_out)
    if os.path.exists(test_out): os.remove(test_out)

    # 2. Load Annotations
    print("Loading annotations...")
    annotations = pd.read_csv(csv_path, sep='\t')
    print(f"Total videos to process: {len(annotations)}")

    # 3. Process each video with a progress bar
    for [att_id, text, _, high, width, length, train] in tqdm(annotations.values, desc="Processing Videos"):
        folder = "train/" if train else "test/"
        path = f"../DATASET/{folder}{att_id}.mp4"
        
        try:
            # Get the list of frames for this video
            extracted_frames = extract_landmarks(path, sample_delay=0.15)
            
            if extracted_frames is None:
                print(f"Skipped {att_id} (Could not open video)")
                continue
                
            # Create the final video-level dictionary
            video_record = {
                "attachment_id": att_id,
                "text": text,           # The Russian Label
                "length": length,
                "frames": extracted_frames
            }
            
            # Route to the correct file based on the 'train' boolean
            target_file = train_out if train else test_out
            
            # Append safely
            with open(target_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(video_record, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"Error processing {att_id}: {e}")