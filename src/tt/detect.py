import cv2
import numpy as np
from tt.frame import Player


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

    boxes_with_confidence = [
        (box, confidence, (box[2] - box[0]) * (box[3] - box[1]))
        for box, confidence in zip(player_boxes, confidences)
    ]
    sorted_boxes = sorted(
        boxes_with_confidence, key=lambda x: (x[2], x[1]), reverse=True
    )

    players: list[Player] = []
    for box in sorted_boxes:
        p = Player(bbox=[int(e) for e in box[0]], confidence=float(box[1]), label=-1)
        players.append(p)

    return players


def filter_gray(frame):
    tolerance = (
        60  # Adjust tolerance based on how strictly you want to detect gray pixels
    )

    # Separate channels
    b, g, r = cv2.split(frame)

    # Calculate absolute difference between channels
    diff_rg = cv2.absdiff(r, g)
    diff_rb = cv2.absdiff(r, b)
    diff_gb = cv2.absdiff(g, b)

    # Create a mask where differences are within the tolerance range
    gray_mask = (diff_rg <= tolerance) & (diff_rb <= tolerance) & (diff_gb <= tolerance)

    # Create a 3-channel mask
    gray_mask_3d = np.stack([gray_mask] * 3, axis=-1)

    # Apply mask to the frame
    gray = np.where(gray_mask_3d, frame, 0)
    return cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)


def detect_balls(frame, prev_frame):
    # Set up parameters for ball detection

    min_radius, max_radius = 1, 10  # Approximate radius range for table tennis ball
    gray = filter_gray(frame)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = filter_gray(prev_frame)

    frame_diff = cv2.absdiff(prev_gray, gray)

    # Threshold the difference to get a binary image
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to reduce noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store detected balls with color information
    detected_balls = []

    for contour in contours:
        # Filter by contour area to find the likely ball
        # area = cv2.contourArea(contour)

        # Get the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)

        # Check if the radius is within the expected range for the ball
        if min_radius <= radius <= max_radius:
            # Extract color information within the detected ball's circle
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
            mean_color = cv2.mean(frame, mask=mask)[:3]

            # Calculate distance to white color (255, 255, 255)
            color_distance_to_white = np.sqrt(sum((c - 255) ** 2 for c in mean_color))

            # Store ball data
            detected_balls.append(
                {
                    "center": (int(x), int(y)),
                    "radius": radius,
                    "color_distance_to_white": color_distance_to_white,
                }
            )

    # Sort detected balls by color distance to white and keep the top 3
    detected_balls = sorted(detected_balls, key=lambda b: b["color_distance_to_white"])[
        :3
    ]
    img = thresh, frame_diff
    return detected_balls, img
