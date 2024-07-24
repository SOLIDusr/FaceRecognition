from deepface import DeepFace as dface
import cv2 as cv
import asyncio


async def frameAnalyze(frame, antiSpoofing: bool):
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

async def stream(capture, antiSpoofing):
    count = 0
    rectData = [[(0, 0), (0, 0), "None"]]
    while capture:
        count += 1
        ret, frame = capture.read()
        if not ret:
            break
        if count % 5 == 0:
            collected = await frameAnalyze(frame, antiSpoofing)
            if not collected:
                rectData = [[(0, 0), (0, 0), "None"]]
            else: rectData = collected
        for face in rectData:
            frame = cv.rectangle(frame, face[0], face[1], (255, 0, 0), 2)
            frame = cv.putText(frame, f"{face[2]}", (face[0][0], face[0][1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        cv.imshow("result", frame)
        if cv.waitKey(1) == ord('q'):
            break


def main():
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
