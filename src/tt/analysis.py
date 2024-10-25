import cv2
import os.path as op
import tt
import tt.detect
import tt.draw


def process_video(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Load YOLO
    net = cv2.dnn.readNet(
        op.join(op.dirname(tt.__file__), "data/yolov4.weights"),
        op.join(op.dirname(tt.__file__), "data/yolov4.cfg"),
    )
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    with open(op.join(op.dirname(tt.__file__), "data/coco.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Create the background subtractor object
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        player_boxes, class_ids, confidences = tt.detect.detect_players(frame, net)
        balls = tt.detect.detect_balls(frame, fgbg, player_boxes)

        # frame = detect_table(frame)

        frame = tt.draw.draw_players(
            player_boxes, class_ids, confidences, frame, classes
        )
        frame = tt.draw.draw_balls(balls, frame)

        # Display the frame with detected objects
        cv2.imshow("YOLO with White Ball Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()
