import json
import os

with open("crowddetection.ipynb", "r") as f:
    nb = json.load(f)
    
new_code = """import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def analyze_crowd_distancing_and_heatmaps():
    # ==========================================
    # Configuration
    # ==========================================
    # Try to load the video WITH audio first, fallback to the silent one
    INPUT_VIDEO_PATH = "output_video_with_audio.mp4" if os.path.exists("output_video_with_audio.mp4") else "output_video.mp4"
    OUTPUT_VIDEO_PATH = "augmented_analytics_video.mp4"
    DISTANCE_THRESHOLD = 80 # Pixels - if two people are closer than this, draw a warning line
    
    # ==========================================
    # Initialization
    # ==========================================
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {INPUT_VIDEO_PATH}. Ensure the first step finished running!")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    # Create a blank black image to store all "heat"
    heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

    frame_count = 0
    print("Starting post-processing analytics for Distance and Heatmaps...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Convert frame to HSV to easily find the purely red and green boxes we drew
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the exact RGB/HSV bounds for the Red and Green OpenCV boxes
        # RED mask
        lower_red_1 = np.array([0, 200, 200])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 200, 200])
        upper_red_2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv_frame, lower_red_1, upper_red_1), 
                                  cv2.inRange(hsv_frame, lower_red_2, upper_red_2))
                                  
        # GREEN mask
        lower_green = np.array([50, 200, 200])
        upper_green = np.array([70, 255, 255])
        mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
        
        # Combine all boxes
        mask_all_boxes = cv2.bitwise_or(mask_red, mask_green)
        
        # Find the contours
        contours, _ = cv2.findContours(mask_all_boxes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        
        # THIS IS THE FIX: Create a blank slate of heat for just THIS specific video frame
        current_frame_heat = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
                centers.append((cx, cy))
                cv2.circle(current_frame_heat, (cx, cy), 25, (1), -1)

        # ADD this single frame's heat onto the master accumulator
        heatmap_accumulator += current_frame_heat

        # Draw Social Distancing lines (Purple) between any two people standing too close
        close_pairs = 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                c1, c2 = centers[i], centers[j]
                distance = math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
                if distance < DISTANCE_THRESHOLD:
                    cv2.line(frame, c1, c2, (255, 0, 255), 3)
                    close_pairs += 1
                    
        # Add a text overlay measuring proximity limits
        if close_pairs > 0:
            cv2.putText(frame, f"Proximity Warnings: {close_pairs}", (50, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                        
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"Polished {frame_count} frames...")

    cap.release()
    out.release()
    print(f"\\nVideo Complete! Saved augmented video to: {OUTPUT_VIDEO_PATH}")

    # ==========================================
    # Post-Processing: Restore Audio
    # ==========================================
    print("Attempting to restore audio track to augmented analytics video...")
    try:
        from moviepy import VideoFileClip
        if os.path.exists(INPUT_VIDEO_PATH):
            original_clip = VideoFileClip(INPUT_VIDEO_PATH)
            if original_clip.audio is not None:
                augmented_clip = VideoFileClip(OUTPUT_VIDEO_PATH)
                final_clip = augmented_clip.with_audio(original_clip.audio)
                
                final_audio_path = OUTPUT_VIDEO_PATH.replace(".mp4", "_with_audio.mp4")
                final_clip.write_videofile(final_audio_path, codec="libx264", audio_codec="aac", logger=None)
                print(f"\\nSuccess! Saved final analytics video WITH audio to: {final_audio_path}")
                
                original_clip.close()
                augmented_clip.close()
                final_clip.close()
            else:
                print("No active audio track found. Proceeding without audio substitution.")
                original_clip.close()
    except ImportError:
        print("Note: moviepy not installed, skipping audio multiplexing.")
    except Exception as e:
        print(f"Failed to process audio restoration: {e}")

    # ==========================================
    # Finalize and Save the Heatmap Image
    # ==========================================
    print("\\nNormalizing and Blurring Heatmap data...")
    blurred_heatmap = cv2.GaussianBlur(heatmap_accumulator, (51, 51), 0)
    heatmap_accum_normalized = cv2.normalize(blurred_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_accum_normalized, cv2.COLORMAP_JET)
    
    cv2.imwrite("heatmap_result.png", heatmap_color)
    print("Heatmap saved to: heatmap_result.png")

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
    plt.title("Crowd Density Heatmap (Red = Heavy Loitering)")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    analyze_crowd_distancing_and_heatmaps()
"""

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell.get("source", []))
        if "def analyze_crowd_distancing_and_heatmaps():" in source_str:
            lines = [line + "\\n" for line in new_code.split("\\n")]
            if lines:
                lines[-1] = lines[-1].replace("\\n", "")
            cell["source"] = lines
            print("Cell successfully updated!")

with open("crowddetection.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
