import numpy as np
from sklearn.linear_model import LinearRegression
import cv2


def calculate_line_length(line):
    """Calculates the length of a line segment."""
    x1, y1, x2, y2 = line[0]
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d


def filter_lines_by_length(lines, min_length):
    """Filters lines based on a minimum length."""
    return [line for line in lines if calculate_line_length(line) >= min_length]


def calculate_mse(line1, line2):
    """Calculates the mean squared error after fitting a regression line through the endpoints."""
    # Get the endpoints of the lines
    (x1, y1, x2, y2) = line1[0]
    (x3, y3, x4, y4) = line2[0]

    # Prepare the data for linear regression
    X = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    y = np.array([y1, y2, y3, y4])

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X[:, 0].reshape(-1, 1), y)

    # Predict y values based on the fitted line
    y_pred = model.predict(X[:, 0].reshape(-1, 1))

    # Calculate mean squared error
    mse = np.mean((y - y_pred) ** 2)
    return mse


def get_color_stats(quadrilateral, frame):
    points = np.array(quadrilateral, dtype=np.int32)

    # Create a mask for the quadrilateral
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    mean_val, std_dev_val = cv2.meanStdDev(frame, mask=mask)
    mean_inside = mean_val.flatten()
    std_dev_inside = std_dev_val.flatten()

    # Calculate mean and standard deviation along the boundary
    # Create an empty mask for the edges and draw the edges only
    edge_mask = np.zeros_like(mask)
    cv2.polylines(edge_mask, [points], isClosed=True, color=255, thickness=1)

    # Calculate mean and standard deviation along the edges
    mean_val_edge, std_dev_val_edge = cv2.meanStdDev(frame, mask=edge_mask)
    mean_along_edge = mean_val_edge.flatten()
    std_dev_along_edge = std_dev_val_edge.flatten()

    def distance_to_white(color):
        white_rgb = np.array([255, 255, 255])
        color_rgb = np.array(color)
        return float(np.linalg.norm(white_rgb - color_rgb))

    return (
        [int(e) for e in list(mean_inside)],
        float(np.linalg.norm(std_dev_inside)),
        [int(e) for e in list(mean_along_edge)],
        float(np.linalg.norm(std_dev_along_edge)),
        distance_to_white(mean_along_edge),
    )


def compute_overlap(box1, box2):
    # Box format: (x1, y1, x2, y2)
    x1_1, y1_1, w_1, h_1 = box1
    x1_2, y1_2, w_2, h_2 = box2
    x2_1 = x1_1 + w_1
    y2_1 = y1_1 + h_1
    x2_2 = x1_2 + w_2
    y2_2 = y1_2 + h_2

    # Calculate the coordinates of the intersection box
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)

    # Check if there is an intersection
    if x1_int < x2_int and y1_int < y2_int:
        # Intersection area
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)

        # Areas of the bounding boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Union area
        union_area = area1 + area2 - intersection_area

        # Percentage of overlap
        overlap_percentage = (intersection_area / union_area) * 100
        return overlap_percentage
    else:
        # No intersection
        return 0.0


def estimate_scale(players) -> float:
    estimated_scale = -1

    player_height_pixels = int(
        np.mean([player.bbox[3] - player.bbox[1] for player in players])
    )
    player_real_height = 170

    estimated_scale = (
        player_real_height / player_height_pixels if player_height_pixels != 0 else -1
    )

    return float(estimated_scale)


def get_moving_pixels(frame, previous_frame, bounding_boxes):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    # Compute optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray_previous, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Calculate the magnitude of optical flow
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold to highlight significant motion
    motion_threshold = 1.0
    motion_mask = (magnitude > motion_threshold).astype(np.uint8) * 255

    # Calculate moving pixels in each bounding box
    moving_pixels_counts = []
    for x, y, w, h in bounding_boxes:
        roi = motion_mask[y : y + h, x : x + w]
        moving_pixels = cv2.countNonZero(roi)
        moving_pixels_counts.append(moving_pixels)
    return moving_pixels_counts
