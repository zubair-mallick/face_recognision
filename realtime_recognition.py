import os
import cv2
import face_recognition
import numpy as np  # ensure numpy is imported

# 1) LOAD & ENCODE KNOWN FACES
known_face_encodings = []
known_face_names = []

known_dir = "known_faces"
for fname in os.listdir(known_dir):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    label = os.path.splitext(fname)[0]
    image = face_recognition.load_image_file(os.path.join(known_dir, fname))
    encs = face_recognition.face_encodings(image)
    if not encs:
        print(f"⚠️  No face found in {fname}; skipping.")
        continue
    known_face_encodings.append(encs[0])
    known_face_names.append(label)
print(f"✅ Loaded {len(known_face_encodings)} known faces.")

# 2) INITIALIZE WEBCAM
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise RuntimeError("Unable to open webcam.")

print("▶️  Starting real-time recognition. Press 'q' to quit.")

# 3) PROCESS FRAMES
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # BGR → RGB
    rgb_small = small_frame[:, :, ::-1]

    # ⚙️ Fix: make array contiguous in memory
    rgb_small = np.ascontiguousarray(rgb_small)  # :contentReference[oaicite:1]{index=1}

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small)
    # Encode faces (named param avoids ambiguity) :contentReference[oaicite:2]{index=2}
    face_encodings = face_recognition.face_encodings(
        rgb_small,
        known_face_locations=face_locations
    )

    # 4) MATCH & DRAW
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_idx = np.argmin(distances)
        name = known_face_names[best_idx] if distances[best_idx] < 0.6 else "Unknown"

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.rectangle(frame, (left, bottom-25), (right, bottom), (0,255,0), cv2.FILLED)
        cv2.putText(frame, name, (left+6, bottom-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5) CLEANUP
video_capture.release()
cv2.destroyAllWindows()