import click
import tt.analysis


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
def process_video(video_path):
    """
    Process the provided video file.

    VIDEO_PATH: Path to the video file (mandatory).
    """
    click.echo(f"Processing video: {video_path}")
    tt.analysis.process_video(video_path)


if __name__ == "__main__":
    process_video()
