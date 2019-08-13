import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model('blackpinkmodel.h5')
dataset = ['Jisoo', 'Lisa']

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    fram150 = cv2.resize(frame, (150, 150))
    fram150 = fram150/255.0

    result = model.predict(fram150[np.newaxis, ...])
    answer = np.argmax(result[0], axis=-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if result[0][answer] >= 0.7:
        for (x, y, w, h) in faces:
            cv2.putText(frame, "{0} - Acc: {1}%".format(dataset[answer], result[0][answer] * 100), (x, y-10), font, 2,(255, 30, 156), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()