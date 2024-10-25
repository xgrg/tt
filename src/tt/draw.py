import cv2
from matplotlib import pyplot as plt
import numpy as np


def draw_players(players, frame):
    # Draw bounding boxes around detected people
    for player in players:
        x, y, w, h = player.bbox
        color = (0, 255, 0)  # Green for "person"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"person {player.confidence:.2f} ({player.label})",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return frame


def draw_quad(quad, frame, color, thickness):
    quadrilateral = np.array(quad.vertices, dtype=np.int32)
    cv2.polylines(
        frame, [quadrilateral], isClosed=True, color=color, thickness=thickness
    )
    for point in quadrilateral:
        cv2.circle(frame, tuple(point), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.putText(
        frame,
        f"{quad.area:.2f} {','.join(quad.codes)}",
        quadrilateral[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        thickness,
    )


def draw_balls(balls, frame):
    for x, y in balls:
        cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    return frame


def draw_segments_and_quadrilateral(segments, intersections):
    # Create a new figure
    plt.figure(figsize=(8, 8))

    # Draw segments and label them with indices
    for i, (x1, y1, x2, y2) in enumerate(segments):
        plt.plot(
            [x1, x2],
            [y1, y2],
            "b-",
            label="Segments"
            if "Segments" not in plt.gca().get_legend_handles_labels()[1]
            else "",
        )
        # Add index label at the midpoint of the segment
        midpoint_x, midpoint_y = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(midpoint_x, midpoint_y, f"{i}", color="blue", fontsize=12, ha="center")

    # If there are exactly 4 intersections, draw the quadrilateral
    if len(intersections) == 4:
        quad_x = [point[0] for point in intersections] + [intersections[0][0]]
        quad_y = [point[1] for point in intersections] + [intersections[0][1]]
        plt.plot(quad_x, quad_y, "r-", linewidth=2, label="Quadrilateral")

    # Mark the intersection points and label them with indices
    for i, point in enumerate(intersections):
        plt.plot(point[0], point[1], "ro")  # red dots for intersections
        # Add index label next to each intersection point
        plt.text(
            point[0],
            point[1],
            f"{i}",
            color="red",
            fontsize=12,
            ha="right",
            va="bottom",
        )

    # Set plot limits and labels
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axhline(0, color="black", linewidth=0.5, ls="--")
    plt.axvline(0, color="black", linewidth=0.5, ls="--")
    plt.grid()
    plt.title("Segments and Quadrilateral Intersections")
    plt.legend()
    plt.show()


def plot(output_fp):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(output_fp)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Iterate over each unique label and plot pixels vs index
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        plt.plot(
            subset["timestamp"], subset["pixels"], marker="o", label=f"Label {label}"
        )

    # Labels and legend
    plt.xlabel("Index")
    plt.ylabel("Pixels")
    plt.title("Pixels as a Function of Index per Label")
    plt.legend()
    plt.grid(True)
    plt.show()
