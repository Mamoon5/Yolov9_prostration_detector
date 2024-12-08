import os
import cv2
import time
import argparse
from flask import Flask, render_template, request, Response, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# Ensure upload and detection directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('runs/detect', exist_ok=True)

# Global model initialization
model = YOLO('yolov9s.pt')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        # Detect file type and process
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        try:
            if file_extension in ['jpg', 'jpeg', 'png']:
                # Image detection
                img = cv2.imread(filepath)
                results = model(img, save=True)
                
                # Find the latest detection
                latest_detection = max(
                    [os.path.join('runs/detect', d) for d in os.listdir('runs/detect')], 
                    key=os.path.getctime
                )
                detected_image = os.path.join(latest_detection, filename)
                
                return render_template('index.html', image_path=detected_image)
            
            elif file_extension == 'mp4':
                # Video detection
                cap = cv2.VideoCapture(filepath)
                out_path = os.path.join('runs/detect', 'output.mp4')
                
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, 30.0, (frame_width, frame_height))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model(frame, save=True)
                    res_plotted = results[0].plot()
                    out.write(res_plotted)
                
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                
                return render_template('index.html', video_path=out_path)
            
            else:
                return "Unsupported file type", 400
        
        except Exception as e:
            return f"Detection error: {str(e)}", 500
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('runs/detect', filename)

def generate_frames(video_path):
    video = cv2.VideoCapture(video_path)
    while True:
        success, frame = video.read()
        if not success:
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames('runs/detect/output.mp4'), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv9 Object Detection Flask App")
    parser.add_argument("--port", default=5000, type=int, help="Port number")
    args = parser.parse_args()
    
    app.run(host="0.0.0.0", port=args.port, debug=True)