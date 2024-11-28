import cv2
import numpy as np
from tt.frame import Player
from loguru import logger


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


def is_ellipsoid(frame, ball):
    (x, y, radius) = ball
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract region of interest (ROI) around the marker
    roi = gray_frame[max(0, y - radius) : y + radius, max(0, x - radius) : x + radius]

    # Detect contours for ellipse fitting
    blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blurred_roi, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = []

    for contour in contours:
        aspect_ratio, minor_axis, major_axis = -1, -1, -1
        if len(contour) >= 5:  # Minimum points required for ellipse fitting
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            aspect_ratio = minor_axis / major_axis if major_axis > 0 else 0
            res.append((aspect_ratio, major_axis, minor_axis))
    return res


def detect_balls(frame, prev_frame):
    # Set up parameters for ball detection

    min_radius, max_radius = 4, 8  # Approximate radius range for table tennis ball
    gray = filter_gray(frame)
    prev_gray = filter_gray(prev_frame)

    frame_diff = cv2.absdiff(prev_gray, gray)

    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to reduce noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger.debug(f"{len(contours)=}")
    # List to store detected balls with color information
    detected_balls = []

    for contour in contours:
        # Get the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        margin_width = 5
        # Check if the radius is within the expected range for the ball
        if min_radius <= radius <= max_radius:
            # Extract color information within the detected ball's circle
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(mask, (int(x), int(y)), radius, 255, -1)

            # Define the expanded mask for the margin area
            expanded_mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(expanded_mask, (int(x), int(y)), radius + margin_width, 255, -1)

            # Calculate the margin mask (expanded mask minus the original mask)
            margin_mask = cv2.subtract(expanded_mask, mask)
            mean_color = cv2.mean(gray, mask=mask)[:3][0]
            mean_color_margin = cv2.mean(gray, mask=margin_mask)[:3][0]

            for aspect_ratio, major_axis, minor_axis in is_ellipsoid(
                frame, (int(x), int(y), radius)
            ):
                detected_balls.append(
                    {
                        "center": (int(x), int(y)),
                        "radius": radius,
                        "mean_color": mean_color,
                        "mean_color_margin": mean_color_margin,
                        "contrast": mean_color / mean_color_margin
                        if mean_color_margin != 0
                        else mean_color / 0.1,
                        "aspect_ratio": aspect_ratio,
                        "major_axis": major_axis,
                        "minor_axis": minor_axis,
                    }
                )

    detected_balls = sorted(detected_balls, key=lambda b: b["contrast"], reverse=True)

    return detected_balls, (thresh, frame_diff)
