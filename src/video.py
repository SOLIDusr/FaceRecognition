import numpy as np
import cv2 as cv
import asyncio
from deepface import DeepFace as dface
from collections import deque
import concurrent.futures


executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)


async def frameAnalyze(frame):
    frame_resized = cv.resize(frame, (320, 240))
    try:
        faces = dface.analyze(frame_resized, actions=("emotion"), enforce_detection=False)
    except ValueError:
        return []
    facesData = []
    for face in faces:
        if face["face_confidence"] < 0.7:
            continue
        scale_x = frame.shape[1] / frame_resized.shape[1]
        scale_y = frame.shape[0] / frame_resized.shape[0]
        x, y, w, h = [int(face["region"][dim] * scale) for dim, scale in zip(("x", "y", "w", "h"), (scale_x, scale_y, scale_x, scale_y))]
        facesData.append([(x, y), (x + w, y + h), face["dominant_emotion"]])
    return facesData


def getMostPopular(emotions: deque) -> str:
    if not emotions:
        return "None"
    counts = {}
    for emotion in emotions:
        counts[emotion] = counts.get(emotion, 0) + 1
    return max(counts, key=counts.get)


async def videoStream(cap):
    count = 0
    rectData = [(0, 0), (0, 0), "None"]
    emotionArray = deque(maxlen=5)
    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if count % 5 == 0:
            facesData = await frameAnalyze(frame)
            if not facesData:
                rectData = [(0, 0), (0, 0), "None"]
            else:
                rectData = facesData[0]
                emotionArray.append(rectData[2])
        frame = cv.rectangle(frame, rectData[0], rectData[1], (255, 0, 0), 2)
        frame = cv.putText(frame, getMostPopular(emotionArray), (rectData[0][0], rectData[0][1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        cv.imshow("result", frame)
        if cv.waitKey(25) == ord('q'):
            break


def main():
    cap = cv.VideoCapture('src/randomhappy.mp4')
    cap.set(cv.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    try:
        asyncio.run(videoStream(cap=cap))
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
        