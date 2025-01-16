import numpy as np
import time
import torch
from scipy import spatial
import cv2
import os
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics kütüphanesi yüklenemedi. Lütfen 'pip install ultralytics' komutunu çalıştırın.")

input_video_path = input("Lütfen işlenmek istenen video dosyasının yolunu girin: ").strip().replace('"', '').replace("'", '')
output_video_path = "output_yolov8_video.avi"

if not os.path.exists(input_video_path):
    raise FileNotFoundError(f"Girdi video dosyası bulunamadı: {input_video_path}. Lütfen geçerli bir dosya yolu girin.")

model = YOLO('yolov8x.pt')

video_stream = cv2.VideoCapture(input_video_path)
video_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_fps = video_stream.get(cv2.CAP_PROP_FPS)
total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(output_video_path, fourcc, source_fps, (video_width, video_height), True)

previous_frame_detections = [{(0, 0): 0} for _ in range(10)]
tracked_objects = {}
object_id_counter = 0

start_time = time.time()

frame_count = 0
while True:
    (grabbed, frame) = video_stream.read()
    if not grabbed:
        break

    frame_count += 1

    results = model(frame)

    boxes, confidences, classIDs, centroids = [], [], [], []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centerX, centerY = (x1 + x2) // 2, (y1 + y2) // 2

            if confidence > 0.5:
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(confidence))
                classIDs.append(class_id)
                centroids.append((centerX, centerY))

    current_frame_centroids = centroids
    current_frame_classIDs = classIDs

    current_detections = {}
    if len(boxes) > 0:
        for i in range(len(boxes)):
            (x, y, w, h) = boxes[i]
            centerX, centerY = (x + w // 2, y + h // 2)
            width, height = w, h

            dist = np.inf
            for frame_num in range(len(previous_frame_detections)):
                coordinate_list = list(previous_frame_detections[frame_num].keys())
                if len(coordinate_list) == 0:
                    continue
                temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
                if temp_dist < dist:
                    dist = temp_dist
                    matched_coord = coordinate_list[index[0]]
                    matched_frame_num = frame_num

            if dist > (max(width, height) / 2):
                current_detections[(centerX, centerY)] = object_id_counter
                object_id_counter += 1
            else:
                current_detections[(centerX, centerY)] = previous_frame_detections[matched_frame_num][matched_coord]

    detected_objects = {}
    for (centerX, centerY), object_id in current_detections.items():
        label = model.names[classIDs[centroids.index((centerX, centerY))]]
        if label in detected_objects:
            detected_objects[label] += 1
        else:
            detected_objects[label] = 1

    for i in range(len(boxes)):
        if classIDs[i] < len(model.names):
            (x, y, w, h) = boxes[i]
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = model.names[classIDs[i]]
            text = f"{label}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vehicle_classes = ["bicycle", "car", "motorbike", "bus", "truck"]
    total_vehicles = sum([count for label, count in detected_objects.items() if label in vehicle_classes])

    if total_vehicles == 0:
        traffic_density = ""
    elif total_vehicles > 14:
        traffic_density = "High Traffic"
        traffic_color = (0, 0, 255)
    elif total_vehicles > 7:
        traffic_density = "Medium Traffic"
        traffic_color = (0, 255, 255)
    else:
        traffic_density = "Low Traffic"
        traffic_color = (0, 255, 0)

    text_x, text_y = 10, 30
    if traffic_density:
        cv2.putText(frame, "Traffic Density: " + traffic_density, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, traffic_color, 2)
    text_y += 25

    detected_objects_text = ""
    for obj, count in detected_objects.items():
        detected_objects_text += f"{obj}: {count}\n"

    for line in detected_objects_text.split('\n'):
        if line:
            cv2.putText(frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            text_y += 20

    writer.write(frame)

    cv2.imshow("Processed Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed_time = time.time() - start_time
    remaining_time = (elapsed_time / frame_count) * (total_frames - frame_count)
    completion_percentage = (frame_count / total_frames) * 100
    print(f"%{completion_percentage:.2f} tamamlandı, kalan süre: {remaining_time / 60:.2f} dakika", end="\r")

    previous_frame_detections.pop(0)
    previous_frame_detections.append(current_detections)

writer.release()
video_stream.release()
cv2.destroyAllWindows()

print(f"\nVideo işlem tamamlandı, çıktı: {output_video_path}")