from tt import polygon, math
import json
import numpy as np


class Player(object):
    bbox: list[int, int, int, int] = []
    label: int = -1
    confidence: float = 0.0
    pixels: int = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Frame(object):
    index: int = 0
    players: list[Player] = []
    polygons: list[polygon.Polygon] = []
    scale: float = 1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_rows(self, fps, start_index):
        rows = []
        for p in self.players:
            row = [
                self.index + start_index,
                f"{(start_index + self.index) / fps:.2f}",
                "player",
            ]
            row.extend([p.label, json.dumps(p.bbox), None, None, None, None, None])
            rows.append(row)

        for p in self.polygons:
            codes = list(p.codes)
            if (
                "NOT_CONVEX" not in p.codes
                and "NOT_QUADRI" not in p.codes
                and "AREA" not in p.codes
                and "BAD_SUM_OF_ANGLES" not in p.codes
                and "EXTREME_ANGLES" not in p.codes
                and "EDGE_RATIO" not in p.codes
            ):
                row = [
                    self.index + start_index,
                    f"{(start_index + self.index) / fps:.2f}",
                    "quad",
                ]

                row.extend(
                    [
                        p.label,
                        json.dumps(p.vertices),
                        json.dumps(codes),
                        json.dumps(p.angles),
                        json.dumps(p.edges),
                        json.dumps(p.color),
                        json.dumps(p.area),
                    ]
                )
                rows.append(row)
        return rows


def are_same_quadrilateral(quad1, quad2, tolerance=1.0):
    """
    Check if two quadrilaterals are the same within a given tolerance.
    Allows for cyclic permutations of points.
    """
    # Convert to numpy arrays for easy manipulation

    quad1 = np.array(quad1.vertices)
    quad2 = np.array(quad2.vertices)

    # Try all cyclic permutations of quad2
    for i in range(len(quad2)):
        permuted_quad2 = np.roll(quad2, shift=i, axis=0)
        if np.allclose(quad1, permuted_quad2, atol=tolerance):
            return True

    return False


def label_quadrilaterals(quadrilaterals, tolerance=10.0):
    """
    Assign labels to quadrilaterals, grouping similar ones under the same label.
    """
    labels = []
    current_label = 0
    label_map = {}

    for i, quad1 in enumerate(quadrilaterals):
        if i in label_map:
            # Already labeled
            labels.append(label_map[i])
            continue

        # Assign a new label
        labels.append(current_label)
        label_map[i] = current_label

        # Compare with the rest to group similar ones
        for j, quad2 in enumerate(quadrilaterals[i + 1 :], start=i + 1):
            if j not in label_map and are_same_quadrilateral(quad1, quad2, tolerance):
                label_map[j] = current_label

        current_label += 1

    return labels


def label_players(players, history: list[Frame] = []) -> list[int]:
    labels: list[dict[int, float]] = []
    for i, p in enumerate(players):
        labels.append({i: 0})
    for frame in history:
        for i, player in enumerate(players):
            bbox = player.bbox
            for player_in_old_frame in frame.players:
                overlap = math.compute_overlap(player_in_old_frame.bbox, bbox)
                # print(
                #     f"{frame.index} ({len(frame.players)}) : {i} {bbox} / {player_in_old_frame.label} {player_in_old_frame.bbox} = {overlap}"
                # )
                labels[i].setdefault(player_in_old_frame.label, 0)
                labels[i][player_in_old_frame.label] += overlap

    labels2 = []
    used_labels = []
    for each in labels:
        maxlab = -1
        maxval = -1
        for k, v in each.items():
            used_labels.append(k)
            if v > maxval:
                maxval = v
                maxlab = k
        if maxval == 0:
            labels2.append(len(used_labels))
        else:
            labels2.append(maxlab)

    return labels2
