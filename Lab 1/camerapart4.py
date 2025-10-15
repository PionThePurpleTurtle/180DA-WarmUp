import cv2
import numpy as np
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape

    rect_size = 200
    x1, y1 = w//2 - rect_size//2, h//2 - rect_size//2
    x2, y2 = w//2 + rect_size//2, h//2 + rect_size//2

    roi = frame[y1:y2, x1:x2]

    pixels = roi.reshape((-1, 3))

    kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)

    color_tuple = tuple(int(c) for c in dominant_color)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.rectangle(frame, (10, 10), (100, 100), color_tuple, -1)

    text = f"BGR: {color_tuple}"
    cv2.putText(frame, text, (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)


    cv2.imshow("Dominant Color Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
