import sys
import os
import numpy as np
import cv2
import time
from tensorflow.keras.applications import xception
from tensorflow.keras.preprocessing import image as keras_image

def load_and_process_image(img):
    img = cv2.resize(img, (299, 299))
    img_array = np.expand_dims(img, axis=0)
    img_array = xception.preprocess_input(img_array)
    return img_array

def predict_deepfake(model, img_array):
    predictions = model.predict(img_array)
    return predictions

def video_to_frames_and_detect(video_path, model, face_net, save_frames=False):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration} seconds")
    
    frame_number = 0
    fake_frame_count = 0
    highest_score = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to a blob for face detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        
        # Process the frame only if a face is detected
        face_detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                face_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                break
        
        if face_detected:
            img_array = load_and_process_image(face)
            predictions = predict_deepfake(model, img_array)
            
            score = predictions[0][0]
            if score > highest_score:
                highest_score = score
            
            if score > 0.0008:
                print(f"Frame {frame_number}: The frame is likely a deepfakexxxxxxxxxxxxxxxxxxxxxxxx with a score of {score}.")
                fake_frame_count += 1
                
                # Save the detected face frame with Unix timestamp in the filename
                if save_frames:
                    timestamp = int(time.time())
                    frame_filename = f"image_{timestamp}.jpg"
                    cv2.imwrite(frame_filename, face)
            else:
                print(f"Frame {frame_number}: The frame is likely authentic with a score of {score}.")
        
        frame_number += 1
    
    cap.release()
    print(f"Processed {frame_number} frames. Detected {fake_frame_count} fake frames.")
    print(f"The highest score encountered was {highest_score}.")

def main(video_path, model_path, face_model_dir, save_frames=False):
    try:
        model = xception.Xception(weights=model_path, include_top=True)
        
        face_proto = os.path.join(face_model_dir, "deploy.prototxt")
        face_model = os.path.join(face_model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
        
        video_to_frames_and_detect(video_path, model, face_net, save_frames)

    except KeyboardInterrupt:
        print("Execution stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python detect_deepfake_video.py video_path model_path face_model_dir [save_frames]")
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = sys.argv[2]
    face_model_dir = sys.argv[3]
    save_frames = len(sys.argv) == 5 and sys.argv[4].lower() == 'true'
    
    main(video_path, model_path, face_model_dir, save_frames)
