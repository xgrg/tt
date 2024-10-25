import cv2
import numpy as np


def detect_players(frame, net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Initialize lists for detected players' bounding boxes
    player_boxes = []
    class_ids = []
    confidences = []

    height, width, _ = frame.shape

    # YOLO object detection
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Process YOLO output
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Class probabilities
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Detect people (COCO class id for person is 0)
            if class_id == 0 and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the player bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                player_boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(player_boxes, confidences, 0.5, 0.4)

    player_boxes = [each for i, each in enumerate(player_boxes) if i in indexes]
    class_ids = [each for i, each in enumerate(class_ids) if i in indexes]
    confidences = [each for i, each in enumerate(confidences) if i in indexes]

    return player_boxes, class_ids, confidences


def detect_balls(frame, fgbg, player_boxes):
    # Apply background subtraction to isolate moving objects
    fgmask = fgbg.apply(frame)

    # Perform morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    balls = []
    for contour in contours:
        if cv2.contourArea(contour) > 1:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the contour is roughly round (width and height are similar)
            if True:  # 0.8 < w / h < 1.2:
                # Check if the detected contour overlaps with any player
                overlapping = False
                for px, py, pw, ph in player_boxes:
                    # Check for overlap using bounding box intersection
                    if not (x + w < px or x > px + pw or y + h < py or y > py + ph):
                        overlapping = True
                        break

                if overlapping:
                    continue  # Skip contours overlapping with players

                # Crop the region of interest (ROI) for color analysis
                roi = frame[y : y + h, x : x + w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Define HSV range for white color
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 25, 255])

                # Create a mask for white color in the ROI
                mask = cv2.inRange(hsv_roi, lower_white, upper_white)
                white_ratio = cv2.countNonZero(mask) / (w * h)

                # If a significant portion of the object is white, consider it a ball
                if white_ratio < 1:  # Adjust this threshold if necessary
                    balls.append((x, y, w, h))
    return balls
