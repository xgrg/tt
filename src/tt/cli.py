import click
from tt import vid
import tt.frame
from loguru import logger
import pandas as pd

import cv2


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


@click.command()
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
def run(
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
