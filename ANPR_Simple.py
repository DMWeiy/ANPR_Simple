import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO

#!Load the Excel file
authorized_data = pd.read_excel("E:\\VS_Code\\Assignment\\Authorized_list.xlsx")

#!Extract authorized license plates into a set for fast lookup
authorized_plates = set(authorized_data["LicensePlate"].str.upper().str.strip())

#!Initialize EasyOCR reader
reader = easyocr.Reader(["en"])

model = YOLO("E:\\VS_Code\\Assignment\\best.pt")

#!Open a video capture object
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://10.62.94.217:4747/video")

cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

#!While the webcam is open
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # Exit loop if unable to read video capture

    # Detect objects in the frame using YOLO
    results = model(frame, classes=[0], line_width=3)

    for result in results:  # Iterate through each frame's result
        for box in result.boxes:  # Access the detected boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Crop the detected license plate region
            license_plate_img = frame[y1:y2, x1:x2]

            # Perform OCR on the cropped region
            ocr_result = reader.readtext(license_plate_img, detail=0, paragraph=True)
            detected_plate = "".join(ocr_result).upper().strip()

            # Check if the detected plate is authorized
            status = (
                "Authorized" if detected_plate in authorized_plates else "Unauthorized"
            )

            # Draw bounding box
            color = (
                (0, 255, 0) if status == "Authorized" else (0, 0, 255)
            )  # Green for authorized, Red for unauthorized
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Write the license plate text inside the bounding box
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(
                detected_plate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )[0]
            text_x = x1  # Align with the left edge of the bounding box
            text_y = y1 - 10  # Place text slightly above the top edge

            # Ensure the text doesn't go out of frame
            if text_y < 10:  # If text goes out of the top frame boundary
                text_y = y1 + text_size[1] + 10  # Move text below the bounding box

            cv2.putText(
                frame,
                detected_plate,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thickness,
            )

            # Print detected plates and their statuses to the console
            print(f"Detected Plate: {detected_plate}, Status: {status}")
            if status == "Authorized":
                print("Open the Door")

            else:
                print("Access Denied,you are not belong here.")

    # Display the frame with the detected number plate and combined text
    cv2.imshow("Number Plate Recognition", frame)

    # Break the loop if ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
