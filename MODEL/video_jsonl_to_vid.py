import cv2
import json
import numpy as np

def visualize_landmarks(video_in, jsonl_in, video_out):
    cap = cv2.VideoCapture(video_in)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (width, height))

    # Hand skeletal structure
    HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    
    # Pose structure for arms (11,12 are shoulders; 13,14 are elbows; 15,16 are wrists)
    ARM_CONN = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]

    with open(jsonl_in, 'r') as f:
        for line in f:
            data = json.loads(line)
            cap.set(cv2.CAP_PROP_POS_FRAMES, data['frame_idx'])
            success, frame = cap.read()
            if not success: break

            # Helper to draw lines and dots
            def draw_skeleton(points, connections, color, thickness=2):
                if not points: return
                for start, end in connections:
                    if start < len(points) and end < len(points):
                        pt1 = (int(points[start]['x']*width), int(points[start]['y']*height))
                        pt2 = (int(points[end]['x']*width), int(points[end]['y']*height))
                        cv2.line(frame, pt1, pt2, color, thickness)
                for p in points:
                    cv2.circle(frame, (int(p['x']*width), int(p['y']*height)), 3, color, -1)

            # Draw Face (Small Blue dots)
            if data['face']:
                for p in data['face']:
                    cv2.circle(frame, (int(p['x']*width), int(p['y']*height)), 1, (255, 0, 0), -1)

            # Draw Arms/Elbows (Thick White lines)
            draw_skeleton(data['pose'], ARM_CONN, (255, 255, 255), 3)
            
            # Draw Hands (Green and Orange)
            draw_skeleton(data['hand1'], HAND_CONN, (0, 255, 0))
            draw_skeleton(data['hand2'], HAND_CONN, (0, 165, 255))

            cv2.putText(frame, f"Time: {data.get('timestamp_sec', 0)}s", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Visualization saved to {video_out}")

if __name__ == "__main__":
    visualize_landmarks("../DATASET/test/0a2ffece-2832-4011-b656-915f39aa7850.mp4", "output_landmarks.jsonl", "verify_full_body.mp4")