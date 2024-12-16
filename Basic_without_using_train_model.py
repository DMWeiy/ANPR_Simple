import cv2
import numpy as np
import imutils
import easyocr
import pandas as pd


#!Load the Excel file
authorized_data = pd.read_excel("E:\\VS_Code\\Assignment\\Authorized_list.xlsx")

#!Extract authorized license plates into a set for fast lookup
authorized_plates = set(authorized_data["LicensePlate"].str.upper().str.strip())

#!Initialize EasyOCR reader
reader = easyocr.Reader(["en"])

#!Open a video capture object
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://10.62.73.130:4747/video")
#!While the webcam is open
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # Exit loop if unable to read video capture

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise and preserve edges
    # ( diameter =11,sigmaColor = sigmaSpace =17)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection using Canny(image, T_lower, T_upper, aperture_size, L2Gradient)
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours in the edged image
    doit = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(doit)

    # Sort contours by area, keeping only the top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    number_plate_location = None
    # Loop through contours to find the number plate-like contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)  # Approximate contour to polygon
        if len(approx) == 4:  # Look for a rectangle-like contour
            number_plate_location = approx
            break

    if number_plate_location is None:
        continue  # Skip frame if no number plate-like contour is found

    # Create a mask for the detected region
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [number_plate_location], 0, 255, -1)
    new_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Extract bounding box of the detected number plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_plate = gray[x1 : x2 + 1, y1 : y2 + 1]

    # Perform OCR on the cropped number plate
    result = reader.readtext(cropped_plate)

    if result:
        # Extract and combine all text lines into a single line
        detected_text = " ".join([res[-2] for res in result])

        # Check if the detected text is in the authorized list
        if detected_text in authorized_plates:
            print(f"Authorized : {detected_text} - Opening the door!")
            frame = cv2.putText(
                frame,
                "Access Granted",
                (y1, x1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        else:
            print(
                f"Unauthorized : {detected_text} - Access Denied,you are not belong here."
            )
            frame = cv2.putText(
                frame,
                "Access Denied",
                (y1, x1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    frame = cv2.rectangle(frame, (y1, x1), (y2, x2), (0, 255, 0), 3)

    # Display the frame with the detected number plate and combined text
    cv2.imshow("Number Plate Recognition", frame)
    # cv2.imshow("Number Plate Recognition filter", bfilter)
    #cv2.imshow("Number Plate Recognition filter1", edged)
    #cv2.imshow("Number Plate Recognition filter2", new_image)
    # Break the loop if ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
