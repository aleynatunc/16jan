from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = Flask(__name__)

# YOLOv8 modelini yükleyin
model = YOLO('yolov8x.pt')

#konya istanbul yolu için yapıldı. inspect ile sitenin nerelere istek attığı incelendi. 
#.ts uzantılı farklı sitelere istek atması sonucu .m3u8 uzantılı site alındı 
# kaynak url: https://www.konyabuyuksehir.tv/canliyayin_izle/44/

# HLS linkini kullanarak video akışını başlat
hls_url = 'https://content.tvkur.com/l/cggk2cokj84dao908mv0/master.m3u8'
video_stream = cv2.VideoCapture(hls_url)

if not video_stream.isOpened():
    raise Exception("Canlı yayın bağlantısı açılamadı. HLS bağlantısı geçerli olmayabilir.")


def generate_frames():
    while True:
        grabbed, frame = video_stream.read()
        if not grabbed:
            break

        # YOLO modelini kullanarak nesne tespiti yapın
        start_time = time.time()
        results = model(frame)
        end_time = time.time()

        # Tespit edilen nesneleri çizin
        detected_objects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[class_id]
                color = tuple(np.random.randint(0, 255, 3).tolist())

                if confidence > 0.3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    detected_objects.append(f"{label} ({confidence:.2f})")

        # İşlenmiş kareyi video akışına gönder
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Ayrıca tespit edilen nesneleri de konsola yazdırabiliriz
        print(f"Tespit edilen nesneler: {', '.join(detected_objects)} | İşlem süresi: {end_time - start_time:.2f} saniye")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)