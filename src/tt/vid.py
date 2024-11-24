import cv2
import os.path as op
import tt
import tt.frame
from tqdm import tqdm
from tt import math, polygon, detect, draw
from loguru import logger


def process_frame(frame, net, frame_index, prev_frame=None, history=[]):
    # Detection of players

    players = detect.detect_players(frame, net)
    labels_players = tt.frame.label_players(players, history=history)
    for p, label in zip(players, labels_players):
        p.label = label

    # Detection of table and balls

    quadrilaterals, blurred, edges, lines = polygon.detect_quadrilaterals(frame)
    balls, thresh, contours = [], None, None
    if prev_frame is not None:
        logger.info("*** Detecting balls")
        balls, (thresh, contours) = detect.detect_balls(frame, prev_frame)
        for b in balls:
            if b["aspect_ratio"] > 0.5:
                logger.debug(b)
            else:
                logger.info(b)
    else:
        logger.warning("*** Skipping ball detection")
    logger.info(f"{len(quadrilaterals)=}")
    quadrilaterals = [
        e
        for e in quadrilaterals
        if len(
            set(e.codes).intersection(
                [
                    "NOT_QUADRI",
                    "NOT_CONVEX",
                    "BAD_SUM_OF_ANGLES",
                    "EXTREME_ANGLES",
                    "EDGE_RATIO",
                ]
            )
        )
        == 0
    ]
    logger.info(f"{len(quadrilaterals)=}")

    labels = tt.frame.label_quadrilaterals(quadrilaterals)
    logger.info(f"{labels=} {len(quadrilaterals)=}")
    quadrilaterals = [
        next(quad for i, quad in enumerate(quadrilaterals) if labels[i] == each)
        for each in set(labels)
    ]
    logger.info(f"{len(quadrilaterals)=}")
    best_quad = polygon.find_best_quad(quadrilaterals)
    if best_quad:
        best_quad.codes.append("BEST")

    # Scale estimation

    scale = math.estimate_scale(players) if players else -1
    if scale != -1:
        table_width_pixels = 274 / scale
        table_height_pixels = 152 / scale
        area = table_height_pixels * table_width_pixels
        logger.info(
            f"Table width: {table_width_pixels} / height: {table_height_pixels}. Expected table area: {area}"
        )
    else:
        logger.error("Error in estimating scale")

    # Storing frame in history

    f = tt.frame.Frame(
        index=frame_index,
        players=players,
        polygons=quadrilaterals,
        table=best_quad,
        scale=scale,
        balls=balls,
    )
    img = [blurred, edges, lines, thresh, contours]
    return f, img


def render_display(frame, f, img):
    blurred, edges, lines, thresh, contours = img

    frame = draw.draw_players(f.players, frame)
    frame = draw.draw_balls(f.balls, frame)

    valid_quadris = [
        quad
        for quad in f.polygons
        if "NOT_CONVEX" not in quad.codes
        and "NOT_QUADRI" not in quad.codes
        and "AREA" not in quad.codes
        and "BAD_SUM_OF_ANGLES" not in quad.codes
        and "EXTREME_ANGLES" not in quad.codes
        and "EDGE_RATIO" not in quad.codes
    ]
    for quad in valid_quadris:
        draw.draw_quad(quad, frame, (255, 0, 255), 1)
    if f.table is not None:
        draw.draw_quad(f.table, frame, (255, 255, 0), 1)

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            edges,
            f"{i}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )
    cv2.imshow("blurred", blurred)
    cv2.imshow("edges", edges)
    cv2.imshow("TT", frame)
    if thresh is not None:
        cv2.imshow("thresh", thresh)
        cv2.imshow("frame_diff", contours)


def process(
    video_path,
    skip_to_frame=0,
    offscreen=False,
    process_n_frames=0,
    process_every_n=10,
    output=None,
):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_to_frame)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load YOLO
    net = cv2.dnn.readNet(
        op.join(op.dirname(tt.__file__), "data/yolov4.weights"),
        op.join(op.dirname(tt.__file__), "data/yolov4.cfg"),
    )
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    frame_seen, frame_processed = 0, 0

    history: list[detect.Frame] = []
    prev_frame = None

    with tqdm(
        total=(total_frames - skip_to_frame) // process_every_n
        if process_n_frames == 0
        else process_n_frames,
        desc="Processing Frames",
        unit="frame",
    ) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_seen += 1
            if frame_seen % process_every_n != 0:
                continue
            frame_processed += 1

            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

            logger.debug(
                "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            )
            msg = f"Frame #{skip_to_frame + frame_seen} (shape: {frame.shape}) ({frame_processed} out of {process_n_frames if process_n_frames != 0 else total_frames})"
            logger.info(msg)

            f, img = process_frame(frame, net, frame_seen, prev_frame, history)

            history.append(f)
            prev_frame = frame.copy()

            if not offscreen:
                render_display(frame, f, img)

            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely until a key is pressed
            pbar.update(1)

            if key == ord(" "):
                if process_n_frames != 0 and frame_processed >= process_n_frames:
                    break
                else:
                    continue
            elif key == ord("q"):
                break
            if process_n_frames != 0 and frame_processed >= process_n_frames:
                break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()

    return history
