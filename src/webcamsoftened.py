from deepface import DeepFace as dface
import cv2 as cv
import asyncio
from collections import deque

async def frameAnalyze(frame: cv.typing.MatLike, antiSpoofing: bool) -> list:
    frame_resized = cv.resize(frame, (320, 240))
    try:
        faces = dface.analyze(frame_resized, actions=("emotion"), enforce_detection=False, anti_spoofing=antiSpoofing)
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

async def stream(capture: cv.VideoCapture, antiSpoofing: bool) -> None:
    count = 0
    boxData = [(0, 0), (0, 0), "None"]
    emotionArray = deque(maxlen=10)
    while capture:
        ret, frame = capture.read()
        if not ret:
            break
        count += 1
        if count % 5 == 0:
            facesData = await frameAnalyze(frame, antiSpoofing)
            if not facesData:
                boxData = [(0, 0), (0, 0), "None"]
            else:
                boxData = facesData[0]
                emotionArray.append(boxData[2])
        frame = cv.rectangle(frame, boxData[0], boxData[1], (255, 0, 0), 2)
        frame = cv.putText(frame, getMostPopular(emotionArray), (boxData[0][0], boxData[0][1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        cv.imshow("result", frame)
        if cv.waitKey(1) == ord('q'):
            break

def main() -> None:
    antiSpoofing = False
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    try:
        asyncio.run(stream(cap, antiSpoofing))
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()