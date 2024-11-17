import click
from tt import vid

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
    vid.process(
        video_path, skip_to_frame, offscreen, process_n_frames, process_every_n, output
    )
