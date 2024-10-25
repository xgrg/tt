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

    min_radius, max_radius = 3, 10  # Approximate radius range for table tennis ball
    gray = filter_gray(frame)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = filter_gray(prev_frame)

    # Compute the absolute difference between current and previous frame
    frame_diff = cv2.absdiff(prev_gray, gray)

    # Threshold the difference to get a binary image
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to reduce noise
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_balls = []
    optical_flow_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **optical_flow_params)

    # Compute magnitude and angle of the flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold the magnitude to segment moving regions
    _, motion_mask = cv2.threshold(mag, 1.5, 255, cv2.THRESH_BINARY)
    motion_mask = motion_mask.astype(np.uint8)

    # Apply a brightness threshold to isolate the lighter areas (potential circle)
    _, bright_mask = cv2.threshold(
        gray, 180, 200, cv2.THRESH_BINARY
    )  # Adjust threshold as needed

    # Combine motion mask and bright mask
    combined_mask = cv2.bitwise_and(motion_mask, bright_mask)

    # Find contours to identify possible circular shapes
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 4: Draw detected circles on the original image for visualization
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # for contour in contours:
    #     # Filter by contour area to find the likely ball
    #     # area = cv2.contourArea(contour)
    #     # if area < min_area:
    #     #     continue

    #     # Get the minimum enclosing circle
    #     (x, y), radius = cv2.minEnclosingCircle(contour)
    #     radius = int(radius)

    #     # Check if the radius is within the expected range for the ball
    #     if min_radius <= radius <= max_radius:
    #         # Extract color information within the detected ball's circle
    #         mask = np.zeros(frame.shape[:2], dtype="uint8")
    #         cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
    #         mean_color = cv2.mean(frame, mask=mask)[:3]

    #         # Calculate distance to white color (255, 255, 255)
    #         color_distance_to_white = np.sqrt(sum((c - 255) ** 2 for c in mean_color))

    #         # Store ball data
    #         detected_balls.append({
    #             "center": (int(x), int(y)),
    #             "radius": radius,
    #             "color_distance_to_white": color_distance_to_white
    #         })

    # # Sort detected balls by color distance to white and keep the top 3
    # detected_balls = sorted(detected_balls, key=lambda b: b["color_distance_to_white"])[:1]

    # balls = []
    # # If circles are detected, filter and draw them
    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #     for (x, y, radius) in circles:
    #         # Confirm if the detected circle is a likely ball candidate by color
    #         mask = np.zeros(frame.shape[:2], dtype="uint8")
    #         cv2.circle(mask, (x, y), radius, 255, -1)
    #         mean_color = cv2.mean(frame, mask=mask)

    #         cv2.circle(frame, (x,y), radius=radius, color=(255, 0, 255), thickness=1)
    #         cv2.putText(
    #                     frame,
    #                     f"{mean_color}",
    #                     (x,y-5),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.6,
    #                     (255,0,255),
    #                     1,
    #                 )

    #         # Check if the mean color is within the expected ball color range
    #         if (ball_color_lower[0] <= mean_color[0] <= ball_color_upper[0] and
    #             ball_color_lower[1] <= mean_color[1] <= ball_color_upper[1] and
    #             ball_color_lower[2] <= mean_color[2] <= ball_color_upper[2]):
    #             balls.append((x,y))
    return detected_balls, thresh, motion_mask, output, gray
