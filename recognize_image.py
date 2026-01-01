# import libraries
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="input image path")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("Loading Face Detector...")
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(
    ["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"]
)
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# load the image
image = cv2.imread(args["image"])
if image is None:
    raise ValueError(f"Could not read image: {args['image']}")

image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0),
    swapRB=False,
    crop=False,
)

# detect faces
detector.setInput(imageBlob)
detections = detector.forward()

# loop over detections
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
            continue

        faceBlob = cv2.dnn.blobFromImage(
            face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
        )
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(
            image,
            text,
            (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
        )

# ✅ SAVE RESULT INSTEAD OF SHOWING
os.makedirs("output", exist_ok=True)
output_path = "output/result.png"
cv2.imwrite(output_path, image)

print(f"[INFO] Result saved to {output_path}")
