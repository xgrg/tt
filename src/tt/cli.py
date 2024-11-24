import click
from tt import vid
import tt.frame
from tt import draw
from loguru import logger
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2


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
    "--offscreen", is_flag=True, help="Enable offscreen processing mode (optional)."
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    help="Path to the output file (optional).",
)
def analyze(
    video_path, skip_to_frame, offscreen, process_n_frames, process_every_n, output
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
        video_path, skip_to_frame, offscreen, process_n_frames, process_every_n
    )

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
    logger.success("Video successfully processed.")
    print(data)
    if output:
        data.to_csv(output, index=False)


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
    assert max_frame_idx < total_frames and min_frame_idx < total_frames

    with tqdm(total=max_frame_idx, desc="Processing Frames", unit="frame") as pbar:
        # Process only the relevant segment
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            frame = cv2.resize(frame, (frame_width, frame_height))

            if frame_idx < min_frame_idx:
                frame_idx += 1
                pbar.update(1)

                continue  # Skip frames before the range

            if frame_idx > max_frame_idx:
                break  # Stop processing after the range

            # Check if the current frame index exists in history
            if frame_idx in history_dict:
                data = history_dict[frame_idx]
                table_polygon = data.table

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
