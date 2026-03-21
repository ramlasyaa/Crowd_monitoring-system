import cv2
import requests
import base64
import time
import csv
import os

# Removed polygon functions as we monitor the entire frame
def infer_frame(frame, api_key, model_id):
    """Sends a frame to Roboflow API using standard requests to bypass SDK issues."""
    # Encode frame to JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
    
    url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}"
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    # Try multiple times in case of rate limits or temporary network errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, data=img_base64, headers=headers)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Request Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    return None

def process_video():
    # ==========================================
    # Configuration
    # ==========================================
    ROBOFLOW_API_KEY = "Bze8VXhWTBZHiRDeKex5"
    MODEL_ID = "head-detection-gun9q-mah4d/1"
    
    INPUT_VIDEO_PATH = "/Users/ramlasya/Documents/Crowd_monitoring_robot/crowd_robot/video.mp4" 
    OUTPUT_VIDEO_PATH = "/Users/ramlasya/Documents/Crowd_monitoring_robot/crowd_robot/output_video.mp4"
    
    # The class you want to count (you mentioned detecting heads now based on the model ID)
    # The previous instruction was no_hard_hat, but I'll use None to process all detections
    # or specify the exact class name if it's different in this model
    TARGET_CLASS = "head" # Modify this if your class name in Roboflow is different
    
    # Alert Configuration
    # We now use the entire frame as the region of interest
    CROWD_THRESHOLD = 10 # Alert triggers if this many people are detected in the full frame

    # ==========================================
    # Initialization
    # ==========================================
    print("Starting processing with custom requests client (No SDK)...")

    # Open CSV file to collect analytics data
    METRICS_CSV_PATH = "crowd_metrics.csv"
    csv_file = open(METRICS_CSV_PATH, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    # Write the CSV Header
    csv_writer.writerow(['Frame Number', 'Total Detected', 'Alert Triggered'])

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{INPUT_VIDEO_PATH}'.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    print("Starting video processing. This may take a while depending on your network...")
    
    alert_timestamps = []
    last_beep_time = -10.0 # Start negative to allow immediate beeps

    # ==========================================
    # Video Processing Loop
    # ==========================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # We can implement frame skipping if it runs too slow
        # Process 1 out of every 3 frames (~20 FPS effective instead of 60 FPS)
        if frame_count % 3 != 0: 
           # We still write the original un-edited frame to the output 
           # so the video length stays exactly 14 seconds
           out.write(frame)
           continue
            
        print(f"\nProcessing Frame {frame_count}...")
        
        # Run inference using direct HTTP requests
        result = infer_frame(frame, ROBOFLOW_API_KEY, MODEL_ID)
        
        if result is None:
            print("Skipping frame due to API failure.")
            # Still write the unedited frame so the video doesn't break
            out.write(frame)
            continue
            
        predictions = result.get("predictions", [])
        total_count = 0

        # Draw bounding boxes
        for pred in predictions:
            class_name = pred.get('class')
            confidence = pred.get('confidence', 0.0)
            
            # Since you changed models to 'head-detection', count all detected heads
            total_count += 1
                
            x = pred.get('x')
            y = pred.get('y')
            w = pred.get('width')
            h = pred.get('height')
            
            # Convert center x,y to top-left x1,y1 and bottom-right x2,y2
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            # Target is the entire frame, so we just label it as Detected
            color = (0, 0, 255) # Red bounding box for all detected people
            label_text = f"Detected: {confidence:.2f}"
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw the label with confidence score
            cv2.putText(frame, label_text, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check for crowd threshold alert
        alert_triggered = total_count >= CROWD_THRESHOLD
        
        # Save metrics to CSV for analytics
        csv_writer.writerow([frame_count, total_count, alert_triggered])
        csv_file.flush() # Force write immediately so we can view graphs while it processes
        
        # Display alert message on screen and play beep
        if alert_triggered:
            current_time = frame_count / fps
            print(">>> CROWD ALERT! <<< Threshold exceeded! BEEP!")
            cv2.putText(frame, f"ALERT: CROWD DETECTED! ({total_count}/{CROWD_THRESHOLD})", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            # Play a system beep sound for EVERY triggered frame
            os.system('afplay /System/Library/Sounds/Ping.aiff &')
            alert_timestamps.append(current_time)
        else:
            cv2.putText(frame, f"Crowd Count: {total_count} / {CROWD_THRESHOLD}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        print(f"Frame {frame_count}: Detected {total_count} total.")

        # Write the processed frame with bounding boxes to the output video
        out.write(frame)

    cap.release()
    out.release()
    csv_file.close() # Clean up the CSV file handler
    
    print(f"\nProcessing complete! Processed {frame_count} frames.")
    print(f"Saved processed video to: {OUTPUT_VIDEO_PATH}")
    print(f"Saved analytics data to: {METRICS_CSV_PATH}")

    # ==========================================
    # Post-Processing: Embed Audio in Video
    # ==========================================
    if alert_timestamps:
        print(f"\nAdding embedded audio alerts to the final video. Stand by...")
        try:
            from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
            video_clip = VideoFileClip(OUTPUT_VIDEO_PATH)
            
            # Create a list of audio clips for each timestamp
            audio_clips = []
            beep_sound_path = "/System/Library/Sounds/Ping.aiff"
            
            for t in alert_timestamps:
                # Add beep sound at specific start times
                if t < video_clip.duration:
                    try:
                        beep = AudioFileClip(beep_sound_path).with_start(t)
                        audio_clips.append(beep)
                    except Exception as clip_err:
                        print(f"Could not load audio clip: {clip_err}")
            
            if audio_clips:
                print(f"Mixing {len(audio_clips)} beep sounds into the video...")
                final_audio = CompositeAudioClip(audio_clips)
                final_video = video_clip.with_audio(final_audio)
                
                # Write to a new file
                AUDIO_OUTPUT_PATH = OUTPUT_VIDEO_PATH.replace('.mp4', '_with_audio.mp4')
                final_video.write_videofile(AUDIO_OUTPUT_PATH, codec="libx264", audio_codec="aac", logger=None)
                print(f"Success! Final video with embedded sound saved to: {AUDIO_OUTPUT_PATH}")
            else:
                print("No valid audio clips to add.")
                
            video_clip.close()
        except Exception as e:
            print(f"Failed to add embedded audio. MoviePy error: {e}")

if __name__ == "__main__":
    process_video()
