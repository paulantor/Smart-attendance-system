import cv2
from insightface.app import FaceAnalysis

print("Loading model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded successfully!")

img = cv2.imread("test_photo.jpg")
print("Image loaded:", img is not None, img.shape if img is not None else "N/A")

faces = app.get(img)
print(f"Faces detected: {len(faces)}")
for f in faces:
    print("Box:", f.bbox, "Confidence:", f.det_score)