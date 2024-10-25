import cv2


def draw_players(player_boxes, class_ids, confidences, frame, classes):
    # Draw bounding boxes around detected people
    for i in range(len(player_boxes)):
        x, y, w, h = player_boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green for "person"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return frame


def draw_balls(balls, frame):
    for x, y, w, h in balls:
        cv2.rectangle(
            frame, (x, y), (x + w, y + h), (255, 0, 0), 2
        )  # Blue for the ball
        cv2.putText(
            frame,
            "White Ball",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    return frame
