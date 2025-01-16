import numpy as np
import torch
import cv2
import os
import time

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics kütüphanesi yüklenemedi. Lütfen 'pip install ultralytics' komutunu çalıştırın.")

input_image_path = input("Lütfen işlenmek istenen görüntü dosyasının yolunu girin: ").strip().replace('"', '').replace("'", '')
output_image_path = "output_yolov8_image.jpg"

if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Girdi görüntü dosyası bulunamadı: {input_image_path}. Lütfen geçerli bir dosya yolu girin.")

model = YOLO('yolov8x.pt')

image = cv2.imread(input_image_path)
if image is None:
    raise ValueError("Görüntü dosyası yüklenemedi.")

detected_objects = {}

start_time = time.time()
results = model(image, augment=True)
end_time = time.time()

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = box.conf[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[class_id]
        color = tuple(np.random.randint(0, 255, 3).tolist())

        if confidence > 0.3:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label in detected_objects:
                detected_objects[label] += 1
            else:
                detected_objects[label] = 1

vehicle_classes = ["bicycle", "car", "motorbike", "bus", "truck"]
total_vehicles = sum([count for label, count in detected_objects.items() if label in vehicle_classes])

if total_vehicles > 20:
    traffic_density = "High Traffic"
    traffic_color = (0, 0, 255)  
elif total_vehicles > 10:
    traffic_density = "Medium Traffic"
    traffic_color = (0, 255, 255)  
else:
    traffic_density = "Low Traffic"
    traffic_color = (0, 255, 0)  

text_x, text_y = 10, 20
cv2.putText(image, "Traffic Density: " + traffic_density, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, traffic_color, 2)
text_y += 15

detected_objects_text = ""
for obj, count in detected_objects.items():
    detected_objects_text += f"{obj}: {count}\n"

for line in detected_objects_text.split('\n'):
    if line:
        cv2.putText(image, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        text_y += 15

cv2.imwrite(output_image_path, image)
print(f"Görüntü işleme tamamlandı. Çıktı: {output_image_path}")
print(f"Toplam işlem süresi: {end_time - start_time:.2f} saniye")
cv2.imshow("Processed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()