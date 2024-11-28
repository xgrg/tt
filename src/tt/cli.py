import click
from tt import vid
import tt.frame
from tt import draw
from loguru import logger
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import math


@click.group()
def cli():
    """
    A command-line interface for table tennis video processing and regeneration.
    """
    pass


def get_video_fps(video_path: str) -> float:
    """
    Get the frames per second (FPS) of a video.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        float: Frames per second (FPS) of the video.
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Cannot open the video file: {video_path}")

    # Get the FPS
    fps = video.get(cv2.CAP_PROP_FPS)

    # Release the video resource
    video.release()

    return fps


@cli.command(name="analyze")
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--skip_to_frame", type=int, default=0, help="Frame number to skip to (optional)."
)
@click.option(
    "--process_n_frames",
    type=int,
    default=0,
    help="Stop once processed n frames (default: until end of the video)",
)
@click.option(
    "--process_every_n",
    type=int,
    default=10,
    help="Process every n only (default:10)",
)
@click.option(
    "--save_every_n",
    type=int,
    default=10,
    help="Save every n (default:10)",
)
@click.option(
    "--offscreen", is_flag=True, help="Enable offscreen processing mode (optional)."
)
@click.option("--skip_table", is_flag=True, help="Skip table detection.")
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    help="Path to the output file (optional).",
)
def analyze(
    video_path,
    skip_to_frame,
    offscreen,
    process_n_frames,
    process_every_n,
    output,
    save_every_n,
    skip_table,
):
    """
    Process the provided video file.

    VIDEO_PATH: Path to the video file (mandatory).
    skip_to_frame: Frame number to skip to (optional).
    offscreen: Enable offscreen processing mode (optional).
    """
    if skip_to_frame is not None:
        click.echo(f"Skipping to frame: {skip_to_frame}")

    if offscreen:
        click.echo("Offscreen mode enabled.")

    click.echo(f"Processing video: {video_path}")
    history = vid.process(
        video_path,
        skip_to_frame,
        offscreen,
        process_n_frames,
        process_every_n,
        output,
        save_every_n,
        skip_table,
    )

    logger.success("Video successfully processed.")

    data = []
    all_polygons = [p for f in history for p in f.polygons]
    labels = tt.frame.label_quadrilaterals(all_polygons)
    for p, label in zip(all_polygons, labels):
        p.label = label

    data = pd.concat(
        [
            f.to_dataframe(fps=get_video_fps(video_path), start_index=skip_to_frame)
            for f in history
        ]
    )
    print(data)


@cli.command(name="rebuild")
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("input_video", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    required=True,
    help="Path to the rebuilt video file (mandatory).",
)
def regenerate_video(csv_path, input_video, output):
    """
    Rebuild a video based on a previously built CSV.

    CSV_PATH: Path to the CSV file containing video metadata.
    """
    click.echo(f"Rebuilding video from CSV: {csv_path}")
    click.echo(f"Output file: {output}")

    history = tt.frame.from_csv(csv_path)

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Unable to open input video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for output file
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object
    out = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))

    # Convert history to a dictionary for quick lookup
    history_dict = {item.index: item for item in history}

    # Determine the frame range to process
    frame_indices = sorted(history_dict.keys())
    if not frame_indices:
        print("Error: No valid frame indices in history.")
        cap.release()
        out.release()
        return

    min_frame_idx = frame_indices[0]
    max_frame_idx = frame_indices[-1]

    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame_idx)

    assert max_frame_idx < total_frames and min_frame_idx < total_frames

    with tqdm(
        total=max_frame_idx - min_frame_idx, desc="Processing Frames", unit="frame"
    ) as pbar:
        # Process only the relevant segment
        frame_idx = min_frame_idx
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            frame = cv2.resize(frame, (frame_width, frame_height))

            if frame_idx > max_frame_idx:
                break  # Stop processing after the range

            # Check if the current frame index exists in history
            if frame_idx in history_dict:
                data = history_dict[frame_idx]
                table_polygon = data.table
                if table_polygon is not None:
                    # Extract vertices and draw the polygon
                    vertices = np.array(table_polygon.vertices, dtype=np.int32)
                    cv2.polylines(
                        frame, [vertices], isClosed=True, color=(0, 255, 0), thickness=2
                    )

                frame = draw.draw_balls(data.balls, frame)
                frame = draw.draw_players(data.players, frame)

            # Write the processed frame to the output video
            out.write(frame)
            pbar.update(1)

            frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    print(
        f"Processed video segment ({min_frame_idx} to {max_frame_idx}) saved at: {output}"
    )


@cli.command(name="segment")
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("input_video", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    required=True,
    help="Path to the rebuilt video file (mandatory).",
)
def segment_rallies(csv_path, input_video, output):
    """
    Based on previously detected objects detect rallies from downtime.

    CSV_PATH: Path to the CSV file containing video metadata.
    """
    click.echo(f"Rebuilding video from CSV: {csv_path}")
    click.echo(f"Output file: {output}")

    history = tt.frame.from_csv(csv_path)

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Unable to open input video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for output file
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object
    out = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))

    # Convert history to a dictionary for quick lookup
    history_dict = {item.index: item for item in history}

    # Determine the frame range to process
    frame_indices = sorted(history_dict.keys())
    if not frame_indices:
        print("Error: No valid frame indices in history.")
        cap.release()
        out.release()
        return

    min_frame_idx = frame_indices[0]
    max_frame_idx = frame_indices[-1]
    assert max_frame_idx < total_frames and min_frame_idx < total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame_idx)

    table = [(177.0, 317.0), (278.0, 259.0), (667.0, 265.0), (741.0, 329.0)]
    long_axis, wide_axis = calculate_table_axes_sorted(table)
    logger.debug(f"{long_axis=} {wide_axis=}")
    frame_idx = min_frame_idx
    prev_frame_idx = frame_idx

    with tqdm(
        total=max_frame_idx - min_frame_idx, desc="Processing Frames", unit="frame"
    ) as pbar:
        # Process only the relevant segment

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            frame = cv2.resize(frame, (frame_width, frame_height))

            if frame_idx > max_frame_idx:
                break  # Stop processing after the range

            # Check if the current frame index exists in history

            if frame_idx in history_dict:
                data = history_dict[frame_idx]

                if data.table is not None:
                    # Extract vertices and draw the polygon
                    vertices = np.array(data.table.vertices, dtype=np.int32)
                    cv2.polylines(
                        frame, [vertices], isClosed=True, color=(0, 255, 0), thickness=2
                    )

                frame = draw.draw_balls(data.balls, frame)
                frame = draw.draw_players(data.players, frame)

                if frame_idx > min_frame_idx and prev_frame_idx in history_dict:
                    for label in [2, 3]:
                        v = measure_vector_players(
                            label, history_dict[prev_frame_idx], history_dict[frame_idx]
                        )
                        if v is not None:
                            coords = project_vector_onto_axes(v, long_axis, wide_axis)
                            logger.debug(f"{frame_idx=} {label=} {v=} {coords=}")
                    v = measure_vector_ball(
                        history_dict[prev_frame_idx], history_dict[frame_idx]
                    )
                    if v is not None:
                        coords = project_vector_onto_axes(v, long_axis, wide_axis)
                        logger.warning(f"{frame_idx=} BALL {v=} {coords=}")
                    prev_frame_idx = frame_idx

            # Write the processed frame to the output video
            out.write(frame)
            pbar.update(1)
            frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    print(
        f"Processed video segment ({min_frame_idx} to {max_frame_idx}) saved at: {output}"
    )


def calculate_table_axes_sorted(quadrilateral):
    # Extract the points
    p1, p2, p3, p4 = quadrilateral

    # Calculate the two vectors
    vector1 = (
        (p2[0] + p3[0]) / 2 - (p1[0] + p4[0]) / 2,
        (p2[1] + p3[1]) / 2 - (p1[1] + p4[1]) / 2,
    )

    vector2 = (
        (p4[0] + p3[0]) / 2 - (p1[0] + p2[0]) / 2,
        (p4[1] + p3[1]) / 2 - (p1[1] + p2[1]) / 2,
    )

    # Calculate the lengths of the vectors
    length1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    length2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Sort the vectors by length: longest is table length, shortest is table width
    if length1 >= length2:
        table_length, table_width = vector1, vector2
    else:
        table_length, table_width = vector2, vector1

    return table_length, table_width


def measure_vector_players(player_label, prev_frame, frame):
    label_found = False
    for p in prev_frame.players:
        if p.label == player_label:
            bbox1 = p.bbox
            label_found = True
            break

    if not label_found:
        return None
    label_found = False
    for p in frame.players:
        if p.label == player_label:
            bbox2 = p.bbox
            label_found = True
            break
    if not label_found:
        return None
    logger.info(f"{bbox1=} {bbox2=}")
    center1 = (bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2)
    center2 = (bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2)
    return (center2[0] - center1[0], center2[1] - center1[1])


def measure_vector_ball(prev_frame, frame):
    ball_found = False
    for b in prev_frame.balls:
        if b["aspect_ratio"] > 0.4 and b["contrast"] > 9:
            center1 = b.center
            ball_found = True
            break

    if not ball_found:
        return None
    ball_found = False
    for b in frame.balls:
        if b["aspect_ratio"] > 0.4 and b["contrast"] > 9:
            center2 = b.center
            ball_found = True
            break
    if not ball_found:
        return None

    logger.info(f"{center1=} {center2=}")
    return (center2[0] - center1[0], center2[1] - center1[1])


def project_vector_onto_axes(center_vector, table_length, table_width):
    # Normalize the table axes
    length_norm = math.sqrt(table_length[0] ** 2 + table_length[1] ** 2)
    width_norm = math.sqrt(table_width[0] ** 2 + table_width[1] ** 2)

    length_unit = (table_length[0] / length_norm, table_length[1] / length_norm)
    width_unit = (table_width[0] / width_norm, table_width[1] / width_norm)

    # Project the center vector onto the axes
    length_coord = center_vector[0] * length_unit[0] + center_vector[1] * length_unit[1]
    width_coord = center_vector[0] * width_unit[0] + center_vector[1] * width_unit[1]

    return length_coord, width_coord


# def analyze_history(history):
#     history_dict = {item.index: item for item in history}
#     table = [(177.0, 317.0), (278.0, 259.0), (667.0, 265.0), (741.0, 329.0)]
#     long_axis, wide_axis = calculate_table_axes_sorted(table)

#     # Determine the frame range to process
#     frame_indices = sorted(history_dict.keys())
#     min_frame_idx = frame_indices[0]
#     max_frame_idx = frame_indices[-1]
#     prev_frame_idx = min_frame_idx
#     for frame_idx in range(min_frame_idx + 1, max_frame_idx):
#         for label in ["1", "2"]:
#             v = measure_vector(label, history_dict[prev_frame_idx], history_dict[frame_idx])
#             coords = project_vector_onto_axes(v, long_axis, wide_axis)
#             logger.debug(f'{frame_idx=} {label=} {v=} {coords=}')

#         prev_frame_idx = frame_idx
