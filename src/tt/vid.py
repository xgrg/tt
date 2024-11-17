import cv2
import os.path as op
import tt
import tt.detect
import tt.draw
import tt.frame
from tqdm import tqdm
from tt import draw, math, polygon, detect
from loguru import logger
import pandas as pd



def process_video(
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

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load YOLO
    net = cv2.dnn.readNet(
        op.join(op.dirname(tt.__file__), "data/yolov4.weights"),
        op.join(op.dirname(tt.__file__), "data/yolov4.cfg"),
    )
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    frame_seen, frame_processed = 0, 0

    history: list[detect.Frame] = []

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
                "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            )
            logger.info(
                f"Frame #{skip_to_frame + frame_seen} (shape: {frame.shape}) ({frame_processed} out of {process_n_frames if process_n_frames != 0 else total_frames})"
            )

            # Detection of players

            players = tt.detect.detect_players(frame, net)
            labels_players = tt.frame.label_players(players, history)
            for p, label in zip(players, labels_players):
                p.label = label

            # Detection of table and balls

            quadrilaterals, blurred, edges, lines = polygon.detect_quadrilaterals(frame)
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
                logger.info(
                    (
                        "Table width:",
                        table_width_pixels,
                        "Table height:",
                        table_height_pixels,
                        "Expected table area:",
                        table_height_pixels * table_width_pixels,
                    )
                )
            else:
                logger.error("Error in estimating scale")

            # Storing frame in history

            f = tt.frame.Frame(
                index=frame_seen,
                players=players,
                polygons=quadrilaterals,
                scale=scale,
            )
            history.append(f)
            previous_frame = frame.copy()

            # Rendering display

            if not offscreen:
                frame = tt.draw.draw_players(players, frame)

                valid_quadris = [
                    quad
                    for quad in quadrilaterals
                    if "NOT_CONVEX" not in quad.codes
                    and "NOT_QUADRI" not in quad.codes
                    and "AREA" not in quad.codes
                    and "BAD_SUM_OF_ANGLES" not in quad.codes
                    and "EXTREME_ANGLES" not in quad.codes
                    and "EDGE_RATIO" not in quad.codes
                ]
                for quad in valid_quadris:
                    draw.draw_quad(quad, frame, (255, 0, 255), 1)
                if best_quad is not None:
                    draw.draw_quad(best_quad, frame, (255, 255, 0), 3)

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

            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely until a key is pressed
            pbar.update(1)

            if key == ord(" "):  # Check if spacebar is pressed
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

    if output:
        data = []
        all_polygons = [p for f in history for p in f.polygons]
        labels = tt.frame.label_quadrilaterals(all_polygons)
        for p, l in zip(all_polygons, labels):
            p.label = l

        for f in history:
            data.extend(f.to_rows(fps=fps, start_index=skip_to_frame))

        pd.DataFrame(
            data,
            columns=[
                "index",
                "timestamp",
                "type",
                "label",
                "vertices",
                "codes",
                "angles",
                "edges",
                "color",
                "area",
            ],
        ).to_csv(output, index=False)
