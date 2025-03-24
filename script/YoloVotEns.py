import torch
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion  # Install: pip install ensemble-boxes
import cv2

# Load YOLO models
models = [
    YOLO("yolov8.pt"),  # Replace with actual path to your YOLOv8 model
    YOLO("yolov9.pt"),  # Replace with YOLOv9 weight path
    YOLO("yolov11.pt"), # Replace with YOLOv11 weight path
    YOLO("yolov12.pt")  # Replace with YOLOv12 weight path
]

def run_ensemble_inference(image_path, models, iou_thr=0.5, skip_box_thr=0.001):
    image = cv2.imread(image_path)  # Load image
    height, width = image.shape[:2]

    all_boxes, all_scores, all_labels = [], [], []

    # Run inference on all models
    for model in models:
        results = model(image)  # Perform inference

        boxes, scores, labels = [], [], []
        for r in results:
            for box in r.boxes.xyxy:  # Bounding boxes
                x1, y1, x2, y2 = box.cpu().numpy()
                boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])  # Normalize boxes
            scores.extend(r.boxes.conf.cpu().numpy())  # Confidence scores
            labels.extend(r.boxes.cls.cpu().numpy().astype(int))  # Class labels

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    # Apply Weighted Box Fusion (WBF)
    final_boxes, final_scores, final_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels, weights=[1,1,1,1], iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )

    # Rescale bounding boxes back to original image size
    final_boxes = np.array(final_boxes) * [width, height, width, height]

    # Draw boxes on image
    for (x1, y1, x2, y2), score, label in zip(final_boxes, final_scores, final_labels):
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"Class {label}: {score:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save or display image
    cv2.imwrite("ensemble_result.jpg", image)
    cv2.imshow("Ensemble Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run inference on an image
run_ensemble_inference("test_image.jpg", models)
