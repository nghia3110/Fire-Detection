from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename
import threading
import winsound

app = Flask(__name__, static_url_path='/static')
model = YOLO('best.pt')

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store detection results
detection_results = {}

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(filepath, filename):
    """Process video in a separate thread"""
    # Open video file
    cap = cv2.VideoCapture(filepath)
    
    detections_count = 0
    frame_count = 0
    last_boxes = []  # Store last detected boxes to keep them visible
    frames_to_keep = 10  # Number of frames to keep showing the red rectangle
    
    print(f"Video detection started for {filename}. Press 'q' to quit.")
    
    # Get video FPS for normal playback speed
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default to 30 FPS if unable to get FPS
    delay = int(1000 / fps)  # Calculate delay in milliseconds for normal speed
    
    # Process video frame by frame and display
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection on frame (lowered confidence threshold)
        results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
          # Check if fire detected in this frame
        has_fire = len(results[0].boxes) > 0
        
        # Debug: Print number of detections
        if has_fire:
            print(f"Frame {frame_count}: {len(results[0].boxes)} fire(s) detected!")
            detections_count += 1
            # Store current detections with frame counter
            last_boxes = [(box, frames_to_keep) for box in results[0].boxes]
            
            # Play warning sound (beep) when fire is detected
            # Frequency=2000Hz, Duration=200ms (runs in background)
            winsound.Beep(2000, 200)
        
        # Draw boxes from current detection or recent detections
        boxes_to_draw = []
        if has_fire:
            boxes_to_draw = results[0].boxes
        else:
            # Show previous detections for a few frames
            boxes_to_draw = [box for box, count in last_boxes if count > 0]
            # Decrease counter for each box
            last_boxes = [(box, count - 1) for box, count in last_boxes if count > 0]
        
        # Draw custom red rectangles and labels
        for box in boxes_to_draw:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw thick red rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Prepare label text
            label = f'FIRE DETECTED: {conf:.2%}'
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            
            # Draw red background for text
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 15), 
                (x1 + text_width + 10, y1), 
                (0, 0, 255), 
                -1
            )
            
            # Draw white text
            cv2.putText(
                frame, 
                label, 
                (x1 + 5, y1 - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
        
        # Add frame counter to show video is processing
        cv2.putText(
            frame, 
            f'Frame: {frame_count}', 
            (10, frame.shape[0] - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
          # Add status text
        status = f'{len(boxes_to_draw)} Fire(s) Detected' if len(boxes_to_draw) > 0 else 'No Fire Detected'
        status_color = (0, 0, 255) if len(boxes_to_draw) > 0 else (0, 255, 0)
        cv2.putText(
            frame, 
            status, 
            (10, frame.shape[0] - 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            status_color, 
            2
        )
        
        # Display the frame
        cv2.imshow('Fire Detection - Press Q to quit', frame)
        
        # Always use normal playback speed for smooth video
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    
    # Store results for the frontend
    detection_results[filename] = {
        'total_frames': frame_count,
        'frames_with_detections': detections_count
    }
    
    print(f"Detection completed: {detections_count} detections in {frame_count} frames")

def process_webcam():
    """Process webcam in a separate thread"""
    # Try to open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return
    
    print("Webcam detection started. Press 'q' to quit.")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection on frame (lowered confidence threshold)
        results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
          # Debug: Print detections
        if len(results[0].boxes) > 0:
            print(f"Webcam Frame {frame_count}: {len(results[0].boxes)} fire(s) detected!")
            
            # Play warning sound (beep) when fire is detected
            winsound.Beep(2000, 200)
        
        # Draw custom red rectangles and labels for each detection
        for box in results[0].boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw thick red rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Prepare label text
            label = f'FIRE DETECTED: {conf:.2%}'
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            
            # Draw red background for text
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 15), 
                (x1 + text_width + 10, y1), 
                (0, 0, 255), 
                -1
            )
              # Draw white text
            cv2.putText(
                frame, 
                label, 
                (x1 + 5, y1 - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
        
        # Add status text at bottom
        status = f'{len(results[0].boxes)} Fire(s) Detected' if len(results[0].boxes) > 0 else 'No Fire Detected'
        status_color = (0, 0, 255) if len(results[0].boxes) > 0 else (0, 255, 0)
        cv2.putText(
            frame, 
            status, 
            (10, frame.shape[0] - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            status_color, 
            2
        )
        
        # Display the frame
        cv2.imshow('Fire Detection - Press Q to quit', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Start video processing in a separate thread
        thread = threading.Thread(target=process_video, args=(filepath, filename))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Video detection started! Check the OpenCV window. Press Q to quit.',
            'filename': filename
        })
    else:
        return jsonify({'error': 'Invalid file type. Please upload a video file (mp4, avi, mov, mkv, webm)'}), 400

@app.route('/detect_webcam', methods=['POST'])
def detect_webcam():
    # Start webcam processing in a separate thread
    thread = threading.Thread(target=process_webcam)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Webcam detection started! Check the OpenCV window. Press Q to quit.'})

@app.route('/get_results/<filename>', methods=['GET'])
def get_results(filename):
    """Get detection results for a specific video"""
    if filename in detection_results:
        return jsonify(detection_results[filename])
    else:
        return jsonify({'error': 'No results found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
