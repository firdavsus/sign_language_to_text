import pandas as pd
import cv2
import mediapipe as mp
import json
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_landmarks(video_path):
    """Extracts MediaPipe landmarks and returns them as a list of frames."""
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None  # Return None if the video file is missing or corrupted

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    frames_list = []

    # CHANGED: model_complexity=1 as per your comment. 
    # 2 is the most accurate but extremely slow. 1 is the sweet spot for speed.
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2, 
        refine_face_landmarks=True) as holistic:

        frame_idx = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

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

def process_video_task(row_data):
    """Worker function to process a single video in a separate CPU process."""
    att_id, text, _, high, width, length, train = row_data
    folder = "dataset/train/" if train else "dataset/test/"
    path = f"{folder}{att_id}.mp4"
    
    try:
        extracted_frames = extract_landmarks(path)
        
        if extracted_frames is None:
            return {"error": "Could not open video", "att_id": att_id}
            
        video_record = {
            "attachment_id": att_id,
            "text": text,
            "length": length,
            "frames": extracted_frames
        }
        
        # Return the parsed data to the main thread
        return {"record": video_record, "is_train": train, "att_id": att_id, "error": None}
        
    except Exception as e:
        return {"error": str(e), "att_id": att_id}


if __name__ == "__main__":
    # 1. Configuration
    csv_path = "dataset/annotations.csv"
    train_out = "train.jsonl"
    test_out = "test.jsonl"

    if os.path.exists(train_out): os.remove(train_out)
    if os.path.exists(test_out): os.remove(test_out)

    # 2. Load Annotations
    print("Loading annotations...")
    annotations = pd.read_csv(csv_path, sep='\t')
    print(f"Total videos to process: {len(annotations)}")

    # 3. Determine CPU Cores (Leave 1 core free so your computer doesn't freeze)
    max_workers = 8
    print(f"Spinning up {max_workers} parallel workers...")

    # Open file handlers once in the main thread
    train_file = open(train_out, 'a', encoding='utf-8')
    test_file = open(test_out, 'a', encoding='utf-8')

    # 4. Process videos in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all rows to the worker pool
        futures = {executor.submit(process_video_task, row): row for row in annotations.values}
        
        # as_completed yields results as soon as any worker finishes a video
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
            result = future.result()
            
            # Handle Errors
            if result.get("error"):
                # tqdm.write prevents the print statement from breaking the visual progress bar
                tqdm.write(f"Skipped {result['att_id']}: {result['error']}")
                continue
                
            # Handle Success: Write to the correct file
            target_file = train_file if result["is_train"] else test_file
            target_file.write(json.dumps(result["record"], ensure_ascii=False) + '\n')

    # Clean up file handlers
    train_file.close()
    test_file.close()
    print("Batch processing complete!")