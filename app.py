from flask import Flask, jsonify, render_template, request, Response, send_file, send_from_directory
import os
import cv2
import subprocess
from ultralytics import YOLO
from collections import deque
import numpy as np
from pydantic import BaseModel, Field
from pymongo import MongoClient
from bson import Binary
import requests
import math
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Load YOLOv8 model
model = YOLO("best.pt")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client['road_safety']  # Database name
collection = db['potholes']  # Collection name

# Pydantic Model for Pothole Data
class Pothole(BaseModel):
    longitude: float = Field(..., description="Longitude of the pothole location")
    latitude: float = Field(..., description="Latitude of the pothole location")
    image: bytes = Field(..., description="Pothole image")


# Define font, scale, colors, and position for annotation
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
text_position = (40, 80)
font_color = (0, 0, 0)    # White color for text
background_color = (0, 0, 255)  # Red background for text
backgroundColor={
    "high": (0, 0, 255),
    "medium":(0, 255, 255),
    "low":(0, 255, 0)
}
pastLocation={
    'lat':0,
    'long':0
}


# Initialize deque for smoothing damage percentage
damage_deque = deque(maxlen=5)
cap = cv2.VideoCapture(0)  # Define globally to manage webcam state
running = True  # Control flag for stopping webcam
percentage_damage=0.0 #road damage percentage

# Function to Insert Processed Frame into MongoDB
def insert_pothole_frame(longitude: float, latitude: float, frame):
    # Encode frame as JPEG for efficient storage
    _, encoded_image = cv2.imencode('.jpg', frame)
    image_data = encoded_image.tobytes()

    # Validate and insert data
    pothole_data = Pothole(
        longitude=longitude,
        latitude=latitude,
        image=image_data
    )

    collection.insert_one({
        "longitude": pothole_data.longitude,
        "latitude": pothole_data.latitude,
        "image": Binary(pothole_data.image)
    })

    print(f"Pothole data successfully inserted at ({longitude}, {latitude})")
    
# Function to calculate distance between two coordinates
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in kilometers

    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    distance = R * c
    return distance


# ========== ROUTES ==========
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/image", methods=["GET", "POST"])
def detect_image():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded", 400

        file = request.files["image"]
        filename = file.filename
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        # Run YOLOv8 detection on image
        results = model(img_path)
        result_img_path = os.path.join(RESULT_FOLDER, "processed_" + filename)

        processed_frame = results[0].plot(boxes=False)

        # Initialize damage percentage
        global percentage_damage 
        
        
        print("image init ",percentage_damage)

        # If masks are available, calculate damage percentage
        if results[0].masks is not None:
            total_area = 0
            masks = results[0].masks.data.cpu().numpy()
            image_area = processed_frame.shape[0] * processed_frame.shape[1]  # Total pixels
            

            for mask in masks:
                binary_mask = (mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    total_area += cv2.contourArea(contour)
            
            percentage_damage = (total_area / image_area) * 100
            
        # print("image  ",percentage_damage)

        
        
        # Draw background for text
        cv2.line(processed_frame, (text_position[0], text_position[1] - 10),(text_position[0] + 350, text_position[1] - 10), background_color, 40)

        
        # Put text on image
        cv2.putText(processed_frame, f'Road Damage: {percentage_damage:.2f}%', text_position,font, font_scale, font_color, 2, cv2.LINE_AA)

        # Save processed image with damage percentage
        cv2.imwrite(result_img_path, processed_frame)

        return render_template(
            "image.html",
            input_image=f"/{UPLOAD_FOLDER}/{filename}",
            result_image=f"/{RESULT_FOLDER}/processed_{filename}",
            damage_percentage=f"{percentage_damage:.2f}"
        )

    return render_template("image.html")


@app.route("/video", methods=["GET", "POST"])
def detect_video():
    if request.method == "POST":
        if "video" not in request.files:
            return "No file uploaded", 400

        file = request.files["video"]
        filename = file.filename

        # Allow only .mp4 files
        if not filename.lower().endswith(".mp4"):
            return "Only MP4 files are allowed.", 400

        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)

        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)

        # Define paths for intermediate and final processed videos
        intermediate_video_path = os.path.join(RESULT_FOLDER, f"intermediate_{filename.replace('.mp4', '.avi')}")
        result_video_path = os.path.join(RESULT_FOLDER, f"processed_{filename}")

        # Define the codec and create VideoWriter object (AVI format for OpenCV compatibility)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(intermediate_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # Initialize a deque for smoothing damage percentage
        # if smoothness required use below code
        #damage_deque = deque(maxlen=5)

       

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform YOLO detection
            results = model.predict(source=frame, imgsz=640, conf=0.25)
            processed_frame = results[0].plot(boxes=False)

            # Initialize damage percentage
            global percentage_damage

            # If masks are available, calculate damage percentage
            if results[0].masks is not None:
                total_area = 0
                masks = results[0].masks.data.cpu().numpy()
                image_area = frame.shape[0] * frame.shape[1]  # Total pixels

                for mask in masks:
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        total_area += cv2.contourArea(contour)

                percentage_damage = (total_area / image_area) * 100
            

            '''
            #smoothness is off to enable modify percentage_damage by smoothness_percentage_damage
            
            # Smooth the damage percentage
        
            damage_deque.append(percentage_damage)
            smoothed_percentage_damage = sum(damage_deque) / len(damage_deque)
            '''
        
            '''
            # Draw background line for text
            cv2.line(processed_frame, (text_position[0], text_position[1] - 10),
                     (text_position[0] + 350, text_position[1] - 10), background_color, 40)

            # Annotate damage percentage
            cv2.putText(processed_frame, f'Road Damage: {smoothed_percentage_damage:.2f}%', 
                        text_position, font, font_scale, font_color, 2, cv2.LINE_AA)
            '''
            if percentage_damage>20:
                cv2.line(processed_frame, (text_position[0], text_position[1] - 10),(text_position[0] + 5040, text_position[1] - 10), backgroundColor["high"], 40)
                cv2.putText(processed_frame, f'Road is severely damaged! Vehicle travel is not allowed on this road', text_position,font, font_scale/2, font_color, 2, cv2.LINE_AA)

            elif percentage_damage>5:
                cv2.line(processed_frame, (text_position[0], text_position[1] - 10),(text_position[0] + 342, text_position[1] - 10), backgroundColor["medium"], 40)
                cv2.putText(processed_frame, f'Caution: Rough road ahead. Drive carefully!', text_position,font, font_scale/2, font_color, 2, cv2.LINE_AA)

            else:
                cv2.line(processed_frame, (text_position[0], text_position[1] - 10),(text_position[0] + 372, text_position[1] - 10), backgroundColor["low"], 40)
                cv2.putText(processed_frame, f'Smooth road with minor potholes. Drive safely!', text_position,font, font_scale/2, font_color, 2, cv2.LINE_AA)

            # Write processed frame to intermediate video (AVI format)
            out.write(processed_frame)
            
        # Release video resources
        cap.release()
        out.release()

        # Convert the intermediate AVI video to MP4 with H.264 codec
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", intermediate_video_path,  # Input .avi file
            "-c:v", "libx264",  # Use H.264 codec
            "-crf", "23",  # Quality setting (lower = higher quality)
            "-preset", "fast",  # Speed vs compression tradeoff
            "-c:a", "aac",  # Audio codec
            "-movflags", "faststart",  # Optimize for streaming
            result_video_path  # Output .mp4 file
        ], check=True)

        # Remove intermediate AVI file after conversion
        os.remove(intermediate_video_path)

        return render_template(
            "video.html",
            input_video=f"/{UPLOAD_FOLDER}/{filename}",
            result_video=f"/{RESULT_FOLDER}/processed_{filename}"
        )

    return render_template("video.html")

# Start the webcam
def generate_frames():
    
    global cap  # Use the global cap variable
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Wait until the webcam is running
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model.predict(source=frame, imgsz=640, conf=0.25)
        processed_frame = results[0].plot(boxes=False)
        
        # Initialize damage percentage
        global percentage_damage 

        # If masks are available, calculate damage percentage
        if results[0].masks is not None:
            total_area = 0
            masks = results[0].masks.data.cpu().numpy()
            image_area = frame.shape[0] * frame.shape[1]

            for mask in masks:
                binary_mask = (mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    total_area += cv2.contourArea(contour)

            percentage_damage = (total_area / image_area) * 100

        print("damage percetage is : ",percentage_damage)
        if percentage_damage>1:
            global pastLocation
            # get location and save to DB
            response = requests.get("https://ipinfo.io/json")
            data = response.json()
            currLocationLat,currLocationLong=map(float,data['loc'].split(','))
            if haversine(pastLocation["long"],pastLocation["lat"],currLocationLong,currLocationLat)>0.5:
                insert_pothole_frame(currLocationLat,currLocationLong,processed_frame)
            print(f"Coordinates: {data['loc']}")  # Outputs "latitude,longitude"
            pastLocation["lat"]=currLocationLat
            pastLocation["long"]=currLocationLong

        '''
        #smoothness is off if required modify percentage_damage by smoothed_percentage_damage
        # Smooth the damage percentage
        damage_deque.append(percentage_damage)
        smoothed_percentage_damage = sum(damage_deque) / len(damage_deque)
        '''
        
        # Draw background line for text
        cv2.line(processed_frame, (text_position[0], text_position[1] - 10),
                 (text_position[0] + 350, text_position[1] - 10), background_color, 40)

        # Annotate damage percentage
        cv2.putText(processed_frame, f'Road Damage: {percentage_damage:.2f}%', 
                    text_position, font, font_scale, font_color, 2, cv2.LINE_AA)
        '''
        if percentage_damage>2:
                cv2.line(processed_frame, (text_position[0], text_position[1] - 10),(text_position[0] + 1080, text_position[1] - 10), backgroundColor["high"], 40)
                cv2.putText(processed_frame, f'Road is severely damaged! Vehicle travel is not allowed on this road', text_position,font, font_scale, font_color, 2, cv2.LINE_AA)

        elif percentage_damage>1:
            cv2.line(processed_frame, (text_position[0], text_position[1] - 10),(text_position[0] + 684, text_position[1] - 10), backgroundColor["medium"], 40)
            cv2.putText(processed_frame, f'Caution: Rough road ahead. Drive carefully!', text_position,font, font_scale, font_color, 2, cv2.LINE_AA)

        else:
            cv2.line(processed_frame, (text_position[0], text_position[1] - 10),(text_position[0] + 745, text_position[1] - 10), backgroundColor["low"], 40)
            cv2.putText(processed_frame, f'Smooth road with minor potholes. Drive safely!', text_position,font, font_scale, font_color, 2, cv2.LINE_AA)
        '''
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/webcam')
def web():
    return render_template('webcam.html')  # Load HTML template

@app.route('/video_feed')
def video_feed():
    global running
    running=True
    # print("video feed running : ", running)
    
    if running:
        # Return video stream if running is True
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Return the default image when the webcam is stopped
        return send_file("static/cam.jpg", mimetype="image/jpg")
    
    


@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global running, cap
    running = False  # Stop the webcam stream
    if cap:
        cap.release()  # Release the capture object
        cap = None  # Set cap to None so it can be re-initialized later
    return "Webcam stopped", 200

@app.route("/static/results/<path:filename>")
def serve_result_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)

@app.route('/get_alert' ,methods=["GET", "POST"])
def road_damage_alert():
    global percentage_damage
    print("getAlert ",percentage_damage)
    return jsonify({"damage": percentage_damage})

if __name__ == "__main__":
    app.run(debug=True)
