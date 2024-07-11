import numpy as np
import cv2

min_confidence = 0.6

classes = ["Background", "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car",
           "Cat", "Chair", "Cow", "DiningTable", "Dog", "Horse", "Motorbike", "Person",
           "PottedPlant", "Sheep", "Sofa", "Train", "TvMonitor"]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt.txt", "models/MobileNetSSD_deploy.caffemodel")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[0], frame.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detect_objects = net.forward()

    for i in range(detect_objects.shape[2]):
        confidence = detect_objects[0, 0, i, 2]
        if confidence > min_confidence:
            class_id = int(detect_objects[0, 0, i, 1])

            prediction_text = f"{classes[class_id]} : {confidence:.2f}"
            box = detect_objects[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype('int')

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colors[class_id], 2)

            if start_y > 30:
                y = start_y - 15
            else:
                y = start_y + 15

            cv2.putText(frame, prediction_text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
