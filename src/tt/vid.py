import cv2
import os.path as op
import tt
import tt.frame
from tqdm import tqdm
from tt import math, polygon, detect, draw
from loguru import logger
import pandas as pd


def process_frame(
    frame, net, frame_index, prev_frame=None, history=[], skip_table=False
):
    # Detection of players

    players = detect.detect_players(frame, net)
    labels_players = tt.frame.label_players(players, history=history)
    for p, label in zip(players, labels_players):
        p.label = label

    # Detection of table and balls
    quadrilaterals = []
    if not skip_table:
        quadrilaterals, blurred, edges, lines = polygon.detect_quadrilaterals(frame)
    balls, thresh, frame_diff = [], None, None
    if prev_frame is not None:
        logger.info("*** Detecting balls")
        balls, (thresh, frame_diff) = detect.detect_balls(frame, prev_frame)
        for b in balls:
            if b["aspect_ratio"] > 0.5:
                logger.debug(b)
            else:
                logger.info(b)
    else:
        logger.warning("*** Skipping ball detection")
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

    labels = tt.frame.label_quadrilaterals(quadrilaterals)
    logger.info(f"{labels=} {len(quadrilaterals)=}")
    quadrilaterals = [
        next(quad for i, quad in enumerate(quadrilaterals) if labels[i] == each)
        for each in set(labels)
    ]
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

    if skip_table:
        img = [thresh, frame_diff]
    else:
        img = [blurred, edges, lines, thresh, frame_diff]
    return f, img


def render_display(frame, f, img, skip_table):
    if skip_table:
        [thresh, frame_diff] = img
    else:
        [blurred, edges, lines, thresh, frame_diff] = img

    frame = draw.draw_players(f.players, frame)
    frame = draw.draw_balls(f.balls, frame)

    for quad in polygon.filter_valid(f.polygons):
        draw.draw_quad(quad, frame, (255, 0, 255), 1)
    if f.table is not None:
        draw.draw_quad(f.table, frame, (255, 255, 0), 1)

    if not skip_table:
        edges = draw.draw_lines(lines, edges)
        cv2.imshow("edges", edges)
        cv2.imshow("blurred", blurred)

    cv2.imshow("Output", frame)

    if thresh is not None:
        cv2.imshow("thresh", thresh)
    if frame_diff is not None:
        cv2.imshow("frame_diff", frame_diff)


def process(
    video_path,
    skip_to_frame=0,
    offscreen=False,
    process_n_frames=0,
    process_every_n=10,
    output=None,
    save_every_n=10,
    skip_table=False,
):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_to_frame)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

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
                "=========================================================================================="
            )
            msg = f"Frame #{skip_to_frame + frame_seen} (shape: {frame.shape}) ({frame_processed} out of {process_n_frames if process_n_frames != 0 else total_frames})"
            logger.info(msg)

            f, img = process_frame(
                frame, net, frame_seen, prev_frame, history, skip_table
            )

            history.append(f)
            history = history[-save_every_n:]

            all_polygons = [p for f in history for p in f.polygons]
            labels = tt.frame.label_quadrilaterals(all_polygons)
            for p, label in zip(all_polygons, labels):
                p.label = label

            prev_frame = frame.copy()

            if not offscreen:
                render_display(frame, f, img, skip_table)

            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely until a key is pressed
            pbar.update(1)

            if frame_processed == save_every_n:
                df = pd.concat(
                    [f.to_dataframe(fps, start_index=skip_to_frame) for f in history]
                )
                df.to_csv(output, mode="w", index=False, header=True)
                logger.info(f"Saved in {output}")
            elif frame_processed % save_every_n == 0 or (
                process_n_frames != 0 and frame_processed >= process_n_frames
            ):
                df = pd.concat(
                    [f.to_dataframe(fps, start_index=skip_to_frame) for f in history]
                )
                df.to_csv(output, mode="a", index=False, header=False)
                logger.info(f"Saved in {output}")

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
