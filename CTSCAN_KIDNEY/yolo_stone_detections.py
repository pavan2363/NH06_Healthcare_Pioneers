import os
import json
import cv2
from ultralytics import YOLO

# 1) Load kidney bounding boxes from Labelme JSON
def load_kidney_boxes(json_path):
    with open(json_path) as f:
        data = json.load(f)
    boxes = []
    for shape in data['shapes']:
        points = shape['points']
        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        x_max = max(p[0] for p in points)
        y_max = max(p[1] for p in points)
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes


# 2) Run YOLO detection on one image
def run_yolo_detection(image_path, model):
    results = model.predict(image_path, conf=0.25)
    detections = []
    for r in results:
        for box in r.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append([x_min, y_min, x_max, y_max, conf])
    return detections


# 3) Filter YOLO detections to only those inside kidney boxes
def filter_detections(detections, kidney_boxes):
    filtered = []
    for det in detections:
        x_min, y_min, x_max, y_max, conf = det
        for kbox in kidney_boxes:
            kx_min, ky_min, kx_max, ky_max = kbox
            # Check if detection center is inside kidney box
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            if kx_min <= cx <= kx_max and ky_min <= cy <= ky_max:
                filtered.append(det)
    return filtered


# 4) Main pipeline
def main():
    img_dir = "images"          # your image folder
    ann_dir = "annotations"     # your Labelme JSON folder
    out_dir = "filtered_results"
    os.makedirs(out_dir, exist_ok=True)

    # Load YOLO model
    model = YOLO("best.pt")   # <--- change to your trained kidney stone weights

    for img_file in os.listdir(img_dir):
        if img_file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(img_dir, img_file)
            json_path = os.path.join(ann_dir, img_file.replace(".png", ".json").replace(".jpg", ".json"))

            if not os.path.exists(json_path):
                print(f"No annotation for {img_file}, skipping...")
                continue

            kidney_boxes = load_kidney_boxes(json_path)
            detections = run_yolo_detection(img_path, model)
            filtered = filter_detections(detections, kidney_boxes)

            print(f"{img_file} → {len(filtered)} stones detected inside kidneys")

            # Draw results for visualization
            img = cv2.imread(img_path)
            for (x1, y1, x2, y2, conf) in filtered:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imwrite(os.path.join(out_dir, img_file), img)


if __name__ == "__main__":
    main()
