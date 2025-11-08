from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename
import threading
import winsound
import time
from datetime import datetime

app = Flask(__name__, static_url_path='/static')
model = YOLO('best.pt')

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
FIRE_IMAGES_FOLDER = 'static/fire_images'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FIRE_IMAGES_FOLDER'] = FIRE_IMAGES_FOLDER

# Store detection results
detection_results = {}

# Global variables for single source streaming (video/webcam/single IP camera)
current_frame = None
is_detecting = False
detection_source = None
video_lock = threading.Lock()

# Multi-camera support - for multiple IP cameras
active_cameras = {}  # {camera_id: {'frame': frame, 'is_active': bool, 'ip': str}}
camera_lock = threading.Lock()
camera_counter = 0

# Create folders if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(FIRE_IMAGES_FOLDER):
    os.makedirs(FIRE_IMAGES_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_fire_detection_image(frame, boxes, source_label=""):
    """Save fire detection image with bounding boxes"""
    try:
        # Draw boxes on the frame
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw thick red rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Prepare label text
            label = f'FIRE: {conf:.2%}'
            
            # Draw text background and label
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 15), 
                         (x1 + text_width + 10, y1), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Generate filename with timestamp and source
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        source_prefix = source_label.replace(':', '_').replace('/', '_')[:20] if source_label else ""
        filename = f"fire_{source_prefix}_{timestamp}.jpg" if source_prefix else f"fire_detection_{timestamp}.jpg"
        filepath = os.path.join(FIRE_IMAGES_FOLDER, filename)
        
        # Save image
        cv2.imwrite(filepath, frame)
        print(f"Saved fire detection image: {filename}")
    except Exception as e:
        print(f"Error saving fire image: {e}")

def draw_detection_boxes(frame, boxes_to_draw):
    """Draw detection boxes on frame"""
    for box in boxes_to_draw:
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

def process_video(filepath, filename):
    """Process video in a separate thread"""
    global current_frame, is_detecting, detection_source
    
    is_detecting = True
    detection_source = 'video'
    
    cap = cv2.VideoCapture(filepath)
    
    detections_count = 0
    frame_count = 0
    frame_save_counter = 0
    last_boxes = []
    frames_to_keep = 10
    
    print(f"Video detection started for {filename}.")
    
    # Get video FPS for normal playback speed
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    delay = 1.0 / fps
    
    while cap.isOpened() and is_detecting:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection on frame
        results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
        has_fire = len(results[0].boxes) > 0
        
        if has_fire:
            print(f"Frame {frame_count}: {len(results[0].boxes)} fire(s) detected!")
            detections_count += 1
            last_boxes = [(box, frames_to_keep) for box in results[0].boxes]
            winsound.Beep(2000, 200)
            
            # Save fire detection image every 10 frames
            frame_save_counter += 1
            if frame_save_counter % 10 == 0:
                save_fire_detection_image(frame.copy(), results[0].boxes, filename)
        
        # Draw boxes from current detection or recent detections
        boxes_to_draw = []
        if has_fire:
            boxes_to_draw = results[0].boxes
        else:
            boxes_to_draw = [box for box, count in last_boxes if count > 0]
            last_boxes = [(box, count - 1) for box, count in last_boxes if count > 0]
        
        # Draw detection boxes
        draw_detection_boxes(frame, boxes_to_draw)
        
        # Add frame counter
        cv2.putText(frame, f'Frame: {frame_count}', (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add status text
        status = f'{len(boxes_to_draw)} Fire(s) Detected' if len(boxes_to_draw) > 0 else 'No Fire Detected'
        status_color = (0, 0, 255) if len(boxes_to_draw) > 0 else (0, 255, 0)
        cv2.putText(frame, status, (10, frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Update current frame for streaming
        with video_lock:
            current_frame = frame.copy()
        
        time.sleep(delay)
    
    cap.release()
    
    # Store results
    detection_results[filename] = {
        'total_frames': frame_count,
        'frames_with_detections': detections_count
    }
    
    is_detecting = False
    print(f"Detection completed: {detections_count} detections in {frame_count} frames")

def process_webcam():
    """Process webcam in a separate thread"""
    global current_frame, is_detecting, detection_source
    
    is_detecting = True
    detection_source = 'webcam'
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        is_detecting = False
        return False
    
    print("Webcam detection started.")
    
    frame_count = 0
    frame_save_counter = 0
    last_boxes = []
    frames_to_keep = 10
    
    while is_detecting:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection on frame
        results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
        has_fire = len(results[0].boxes) > 0
        
        if has_fire:
            print(f"Webcam Frame {frame_count}: {len(results[0].boxes)} fire(s) detected!")
            last_boxes = [(box, frames_to_keep) for box in results[0].boxes]
            winsound.Beep(2000, 200)
            
            # Save fire detection image every 10 frames
            frame_save_counter += 1
            if frame_save_counter % 10 == 0:
                save_fire_detection_image(frame.copy(), results[0].boxes, "webcam")
        
        # Draw boxes
        boxes_to_draw = []
        if has_fire:
            boxes_to_draw = results[0].boxes
        else:
            boxes_to_draw = [box for box, count in last_boxes if count > 0]
            last_boxes = [(box, count - 1) for box, count in last_boxes if count > 0]
        
        # Draw detection boxes
        draw_detection_boxes(frame, boxes_to_draw)
        
        # Add status text
        status = f'{len(boxes_to_draw)} Fire(s) Detected' if len(boxes_to_draw) > 0 else 'No Fire Detected'
        status_color = (0, 0, 255) if len(boxes_to_draw) > 0 else (0, 255, 0)
        cv2.putText(frame, status, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Update current frame for streaming
        with video_lock:
            current_frame = frame.copy()
        
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()
    is_detecting = False
    print("Webcam detection stopped.")
    return True

def process_ip_camera_single(ip_address):
    """Process single IP camera (for backward compatibility)"""
    global current_frame, is_detecting, detection_source
    
    is_detecting = True
    detection_source = 'ip_camera'
    
    cap = cv2.VideoCapture(ip_address)
    
    if not cap.isOpened():
        print(f"Error: Could not connect to IP camera at {ip_address}")
        is_detecting = False
        return False
    
    print(f"IP camera detection started for {ip_address}")
    
    frame_count = 0
    frame_save_counter = 0
    last_boxes = []
    frames_to_keep = 10
    
    while is_detecting:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection to IP camera")
            break
        
        frame_count += 1
        
        # Run detection on frame
        results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
        has_fire = len(results[0].boxes) > 0
        
        if has_fire:
            print(f"IP Camera Frame {frame_count}: {len(results[0].boxes)} fire(s) detected!")
            last_boxes = [(box, frames_to_keep) for box in results[0].boxes]
            winsound.Beep(2000, 200)
            
            # Save fire detection image every 10 frames
            frame_save_counter += 1
            if frame_save_counter % 10 == 0:
                save_fire_detection_image(frame.copy(), results[0].boxes, ip_address)
        
        # Draw boxes
        boxes_to_draw = []
        if has_fire:
            boxes_to_draw = results[0].boxes
        else:
            boxes_to_draw = [box for box, count in last_boxes if count > 0]
            last_boxes = [(box, count - 1) for box, count in last_boxes if count > 0]
        
        # Draw detection boxes
        draw_detection_boxes(frame, boxes_to_draw)
        
        # Add status text
        status = f'{len(boxes_to_draw)} Fire(s) Detected' if len(boxes_to_draw) > 0 else 'No Fire Detected'
        status_color = (0, 0, 255) if len(boxes_to_draw) > 0 else (0, 255, 0)
        cv2.putText(frame, status, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Add IP camera label
        cv2.putText(frame, f'IP Camera: {ip_address[:30]}...', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Update current frame for streaming
        with video_lock:
            current_frame = frame.copy()
        
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()
    is_detecting = False
    print("IP camera detection stopped.")
    return True

def process_multi_ip_camera(camera_id, ip_address):
    """Process multiple IP cameras simultaneously with optimized performance"""
    print(f"Starting multi-camera detection for Camera {camera_id}: {ip_address}")
    
    cap = cv2.VideoCapture(ip_address)
    
    if not cap.isOpened():
        print(f"Error: Could not connect to camera {camera_id} at {ip_address}")
        with camera_lock:
            if camera_id in active_cameras:
                active_cameras[camera_id]['is_active'] = False
        return False
      # Optimize camera settings for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
    
    frame_count = 0
    frame_save_counter = 0
    last_boxes = []
    frames_to_keep = 10
    
    with camera_lock:
        if camera_id in active_cameras:
            active_cameras[camera_id]['is_active'] = True
    
    while True:
        # Check if camera should stop
        with camera_lock:
            if camera_id not in active_cameras or not active_cameras[camera_id]['is_active']:
                break
        
        ret, frame = cap.read()
        if not ret:
            print(f"Lost connection to camera {camera_id}")
            break
        
        frame_count += 1
        
        # Run detection on frame (same as original algorithm)
        results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
        has_fire = len(results[0].boxes) > 0
        
        if has_fire:
            print(f"Camera {camera_id} - Frame {frame_count}: {len(results[0].boxes)} fire(s) detected!")
            last_boxes = [(box, frames_to_keep) for box in results[0].boxes]
            
            # Play warning sound (limit frequency to avoid spam)
            if frame_count % 10 == 0:
                try:
                    winsound.Beep(2000, 100)  # Shorter beep for multi-camera
                except:
                    pass
            
            # Save fire detection image every 10 frames
            frame_save_counter += 1
            if frame_save_counter % 10 == 0:
                save_fire_detection_image(frame.copy(), results[0].boxes, f"cam{camera_id}_{ip_address}")
        
        # Draw boxes
        boxes_to_draw = []
        if has_fire:
            boxes_to_draw = results[0].boxes
        else:
            boxes_to_draw = [box for box, count in last_boxes if count > 0]
            last_boxes = [(box, count - 1) for box, count in last_boxes if count > 0]
        
        # Draw detection boxes
        draw_detection_boxes(frame, boxes_to_draw)
        
        # Add camera label
        cv2.putText(frame, f'Camera {camera_id}: {ip_address[:25]}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Add status text
        status = f'{len(boxes_to_draw)} Fire(s)' if len(boxes_to_draw) > 0 else 'No Fire'
        status_color = (0, 0, 255) if len(boxes_to_draw) > 0 else (0, 255, 0)
        cv2.putText(frame, status, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        
        # Update frame in camera dictionary
        with camera_lock:
            if camera_id in active_cameras:
                active_cameras[camera_id]['frame'] = frame.copy()
        
        # Optimize frame rate for multi-camera
        time.sleep(0.033)  # ~30 FPS per camera
    
    cap.release()
    
    with camera_lock:
        if camera_id in active_cameras:
            active_cameras[camera_id]['is_active'] = False
    
    print(f"Camera {camera_id} detection stopped.")
    return True

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/fire')
def fire_detector():
    return render_template('fire_detection.html')

@app.route('/flood')
def flood_detector():
    return render_template('coming_soon.html')

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
            'message': 'Phát hiện video đã bắt đầu! Đang stream video bên dưới.',
            'filename': filename
        })
    else:
        return jsonify({'error': 'Invalid file type. Please upload a video file (mp4, avi, mov, mkv, webm)'}), 400

@app.route('/detect_webcam', methods=['POST'])
def detect_webcam():
    global is_detecting
    
    # Check if webcam is available first
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        test_cap.release()
        return jsonify({'error': 'Không thể truy cập webcam. Vui lòng kiểm tra kết nối webcam của bạn.'}), 400
    test_cap.release()
    
    # Start webcam processing in a separate thread
    thread = threading.Thread(target=process_webcam)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Phát hiện webcam đã bắt đầu! Đang stream video bên dưới.'})

@app.route('/detect_ip_camera', methods=['POST'])
def detect_ip_camera():
    global is_detecting, camera_counter
    
    data = request.get_json()
    ip_address = data.get('ip_address', '')
    
    if not ip_address:
        return jsonify({'error': 'Vui lòng nhập địa chỉ IP camera!'}), 400
    
    # Test connection to IP camera
    test_cap = cv2.VideoCapture(ip_address)
    if not test_cap.isOpened():
        test_cap.release()
        return jsonify({'error': f'Không thể kết nối đến camera IP tại {ip_address}. Vui lòng kiểm tra địa chỉ IP.'}), 400
    test_cap.release()
    
    # Start IP camera processing in a separate thread
    thread = threading.Thread(target=process_ip_camera_single, args=(ip_address,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': f'Phát hiện camera IP đã bắt đầu! Đang stream video bên dưới.'})

@app.route('/add_ip_camera', methods=['POST'])
def add_ip_camera():
    """Add a new IP camera to multi-camera system"""
    global camera_counter
    
    data = request.get_json()
    ip_address = data.get('ip_address', '')
    
    if not ip_address:
        return jsonify({'error': 'Vui lòng nhập địa chỉ IP camera!'}), 400
    
    # Test connection to IP camera
    test_cap = cv2.VideoCapture(ip_address)
    if not test_cap.isOpened():
        test_cap.release()
        return jsonify({'error': f'Không thể kết nối đến camera IP tại {ip_address}. Vui lòng kiểm tra địa chỉ IP.'}), 400
    test_cap.release()
    
    # Create new camera entry
    with camera_lock:
        camera_counter += 1
        camera_id = camera_counter
        active_cameras[camera_id] = {
            'frame': None,
            'is_active': False,
            'ip': ip_address
        }
    
    # Start camera processing in a separate thread
    thread = threading.Thread(target=process_multi_ip_camera, args=(camera_id, ip_address))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': f'Camera IP {camera_id} đã được thêm thành công!',
        'camera_id': camera_id,
        'ip_address': ip_address
    })

@app.route('/remove_ip_camera/<int:camera_id>', methods=['POST'])
def remove_ip_camera(camera_id):
    """Remove an IP camera from multi-camera system"""
    with camera_lock:
        if camera_id in active_cameras:
            active_cameras[camera_id]['is_active'] = False
            del active_cameras[camera_id]
            return jsonify({'message': f'Camera {camera_id} đã được xóa thành công!'})
        else:
            return jsonify({'error': 'Camera không tồn tại!'}), 404

@app.route('/get_active_cameras', methods=['GET'])
def get_active_cameras():
    """Get list of all active cameras"""
    with camera_lock:
        cameras = []
        for cam_id, cam_data in active_cameras.items():
            cameras.append({
                'camera_id': cam_id,
                'ip_address': cam_data['ip'],
                'is_active': cam_data['is_active']
            })
        return jsonify({'cameras': cameras})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global is_detecting, current_frame
    is_detecting = False
    with video_lock:
        current_frame = None
    return jsonify({'message': 'Đã dừng phát hiện.'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route for single source"""
    def generate():
        global current_frame
        while True:
            with video_lock:
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                frame = current_frame.copy()
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed/<int:camera_id>')
def camera_feed(camera_id):
    """Video streaming route for specific IP camera"""
    def generate():
        while True:
            with camera_lock:
                if camera_id not in active_cameras or active_cameras[camera_id]['frame'] is None:
                    time.sleep(0.1)
                    continue
                frame = active_cameras[camera_id]['frame'].copy()
            
            # Encode frame as JPEG with optimization
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_results/<filename>', methods=['GET'])
def get_results(filename):
    """Get detection results for a specific video"""
    if filename in detection_results:
        return jsonify(detection_results[filename])
    else:
        return jsonify({'error': 'No results found'}), 404

@app.route('/get_fire_images', methods=['GET'])
def get_fire_images():
    """Get list of saved fire detection images sorted by newest first"""
    try:
        images = []
        if os.path.exists(FIRE_IMAGES_FOLDER):
            # Get all image files with their modification time
            files_with_time = []
            for filename in os.listdir(FIRE_IMAGES_FOLDER):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(FIRE_IMAGES_FOLDER, filename)
                    file_time = os.path.getmtime(filepath)  # Get modification time
                    files_with_time.append((filename, file_time))
            
            # Sort by modification time (newest first)
            files_with_time.sort(key=lambda x: x[1], reverse=True)
            
            # Build response with sorted images
            for filename, file_time in files_with_time:
                timestamp = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
                images.append({
                    'filename': filename,
                    'timestamp': timestamp
                })
        
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fire_images/<filename>')
def serve_fire_image(filename):
    """Serve fire detection images"""
    return app.send_static_file(f'fire_images/{filename}')

@app.route('/delete_fire_image/<filename>', methods=['DELETE'])
def delete_fire_image(filename):
    """Delete a fire detection image"""
    try:
        filepath = os.path.join(FIRE_IMAGES_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'message': f'Đã xóa ảnh {filename} thành công!'})
        else:
            return jsonify({'error': 'Ảnh không tồn tại!'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
