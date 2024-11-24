import cv2
import numpy as np
from tt.math import get_color_stats
from loguru import logger
import itertools
from tt import math


def rgb_to_hex(rgb_array):
    bgr_int = np.clip(rgb_array, 0, 255).astype(int)
    hex_color = "#{:02X}{:02X}{:02X}".format(bgr_int[2], bgr_int[1], bgr_int[0])
    return hex_color


class Polygon:
    vertices = []
    codes = []
    lines = []
    angles = []
    edges = []
    color = []
    area = -1
    label = -1

    def __str__(self):
        a = f"Polygon ({','.join(self.codes)} - "
        a += f" Vertices ({len(self.vertices)}): {self.vertices} - "
        a += f" Lines: {self.lines} - "
        if len(self.vertices) == 4:
            a += f" Angles: {self.angles[0]:.2f} {self.angles[1]:.2f} {self.angles[2]:.2f} {self.angles[3]:.2f} ({self.angles[0] + self.angles[2]:.2f} {self.angles[1] + self.angles[3]:.2f}) - "
            # a += f" Edges: {self.edges[0]:.2f} {self.edges[1]:.2f} {self.edges[2]:.2f} {self.edges[3]:.2f} - "
            ratios = [
                max(self.edges[0], self.edges[2]) / min(self.edges[0], self.edges[2]),
                max(self.edges[1], self.edges[3]) / min(self.edges[1], self.edges[3]),
            ]
            a += f"Ratios: {ratios[0]:.2f} {ratios[1]:.2f} - "
            a += f"Color: {rgb_to_hex(self.color[0])} {self.color[1]:.2f} - {rgb_to_hex(self.color[2])} {self.color[3]} {self.color[4]}"
        return a

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def detect(
        combo, lines, frame, target_area=30000, tolerance=1000, tolerance_angle=12
    ):
        p = Polygon()
        p.lines = combo

        segments = [lines[e] for e in combo]

        assert len(segments) == 4
        frame_dimensions = frame.shape[:2]

        codes = []
        unique_points = p.find_intersections(
            segments, frame_dimensions[1], frame_dimensions[0]
        )
        if unique_points:
            points = np.array(unique_points, dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            hull = cv2.convexHull(points)
            unique_points = [(int(each[0][0]), int(each[0][1])) for each in hull]
            p.vertices = unique_points
            p.vertices = [tuple([float(e) for e in each]) for each in unique_points]

        else:
            codes.append("NO_INTERSECTION")

        if len(unique_points) != 4:
            codes.append("NOT_QUADRI")
        else:
            # Check angles
            angles = p.get_polygon_angles()
            if (
                abs(angles[0] + angles[2] - 180) > tolerance_angle
                or abs(angles[1] + angles[3] - 180) > tolerance_angle
            ):
                codes.append("BAD_SUM_OF_ANGLES")
            for a in angles:
                if a < 10 or a > 170:
                    codes.append("EXTREME_ANGLES")
                    break

            edges = p.get_polygon_edge_lengths()
            ratios = [
                max(edges[0], edges[2]) / min(edges[0], edges[2]),
                max(edges[1], edges[3]) / min(edges[1], edges[3]),
            ]
            for r in ratios:
                if r > 3:
                    codes.append("EDGE_RATIO")
                    break
            p.angles = angles
            p.edges = [float(e) for e in edges]
            p.color = get_color_stats(unique_points, frame)

        # Check if the quadrilateral is convex
        if not p.is_convex():
            codes.append("NOT_CONVEX")
        area = p.shoelace_area()

        # Check if the area is within the tolerance range of the target area
        if not (target_area - tolerance <= area <= target_area + tolerance):
            codes.append("AREA")

        # Check if all points are within the specified frame dimensions
        for pt in unique_points:
            x, y = pt
            if x < 0 or y < 0 or x >= frame_dimensions[1] or y >= frame_dimensions[0]:
                codes.append("OUT_OF_FRAME")
                break

        p.codes = codes
        p.area = area
        return p

    def find_intersections(self, segments, frame_width, frame_height):
        """Find the intersections of each adjacent pair of segments, including the last and first segments, and filter them."""
        intersections = []

        # Iterate over each adjacent pair of segments
        for i in range(len(segments) - 1):
            intersection = self.line_intersection(segments[i], segments[i + 1])
            if intersection:
                intersections.append(intersection)

        # Check intersection between the last and the first segment
        if len(segments) > 1:
            intersection = self.line_intersection(segments[-1], segments[0])
            if intersection:
                intersections.append(intersection)

        # Filter intersections that are within frame boundaries
        return [
            (x, y)
            for x, y in intersections
            if 0 < x < frame_width and 0 < y < frame_height
        ]

    def shoelace_area(self):
        """Calculates the area of a polygon using the shoelace formula."""
        points = self.vertices
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def line_intersection(self, line1, line2):
        """Calculate the intersection of two lines defined by line1 and line2."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calculate the determinants
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if denom == 0:
            return None  # Lines are parallel or coincident

        # Calculate intersection coordinates
        intersect_x = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denom
        intersect_y = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denom

        return (intersect_x, intersect_y)

    def get_polygon_edge_lengths(self):
        """Calculate the lengths of all edges of a polygon defined by its vertices."""
        lengths = []
        vertices = self.vertices
        num_vertices = len(vertices)

        for i in range(num_vertices):
            # Get the current vertex and the next vertex (wrapping around)
            current_vertex = np.array(vertices[i])
            next_vertex = np.array(
                vertices[(i + 1) % num_vertices]
            )  # Wraps around to the first vertex

            # Calculate the length of the edge
            length = np.linalg.norm(next_vertex - current_vertex)
            lengths.append(length)

        return [float(e) for e in lengths]

    def is_convex(self):
        def cross_product_orientation(p1, p2, p3):
            # Calculate the cross product of the vectors (p2 - p1) and (p3 - p2)
            return (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

        polygon = self.vertices
        n = len(polygon)
        if n < 4:  # A triangle is always convex
            return True

        # Initialize the sign of the first turn
        first_sign = None

        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            p3 = polygon[(i + 2) % n]

            cross_product = cross_product_orientation(p1, p2, p3)

            if cross_product != 0:  # Ignore collinear points
                current_sign = np.sign(cross_product)
                if first_sign is None:
                    first_sign = current_sign
                elif current_sign != first_sign:
                    return False

        return True

    def get_polygon_angles(self):
        polygon = self.vertices
        assert len(polygon) == 4

        def calculate_angle(p1, p2, p3):
            # Vectors p1->p2 and p2->p3
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            # Calculate the angle in radians using the dot product
            dot_product = np.dot(v1, v2)
            mag_v1 = np.linalg.norm(v1)
            mag_v2 = np.linalg.norm(v2)

            # Prevent division by zero
            if mag_v1 == 0 or mag_v2 == 0:
                return 0.0

            angle_rad = np.arccos(dot_product / (mag_v1 * mag_v2))
            return np.degrees(angle_rad)

        # Calculate angles at each corner of the quadrilateral
        angles = []
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i - 1]
            p2 = polygon[i]
            p3 = polygon[(i + 1) % n]
            angle = calculate_angle(p1, p2, p3)
            angles.append(angle)

        return [float(a) for a in angles]


def unique_rotational_combinations(lines, size=4):
    """Generate unique combinations of segments of a given size, ignoring rotations."""
    unique_combinations = []

    for combo in itertools.combinations(lines, size):
        unique_combinations.append(combo)
        unique_combinations.append([combo[0], combo[2], combo[1], combo[3]])
        unique_combinations.append([combo[0], combo[2], combo[3], combo[1]])

    return unique_combinations


def erase_border(frame, border_percentage=15):
    height, width = frame.shape[:2]

    border_size_h = int(height * (border_percentage / 100))
    border_size_w = int(width * (border_percentage / 100))

    frame[:border_size_h, :] = 0
    frame[-border_size_h:, :] = 0
    frame[:, :border_size_w] = 0
    frame[:, -border_size_w:] = 0

    return frame


def detect_quadrilaterals(frame):
    # Preprocessing
    blurred = cv2.bilateralFilter(frame, d=51, sigmaColor=50, sigmaSpace=125)
    blurred = cv2.fastNlMeansDenoisingColored(
        blurred, h=10, templateWindowSize=11, searchWindowSize=31
    )
    blurred = cv2.medianBlur(blurred, ksize=5)
    edges = cv2.Canny(blurred, 50, 150)
    edges = erase_border(edges)

    # Line detection
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=70, minLineLength=5, maxLineGap=100
    )
    logger.info("Lines detected: %s" % len(lines))

    lines = merge_lines(lines)
    logger.info("Merged lines: %s" % len(lines))

    logger.info(dict([(i, line) for (i, line) in enumerate(lines)]))

    # Quadrilateral detection
    combinats = list(unique_rotational_combinations(list(range(len(lines))), 4))
    logger.info(f"Testing {len(combinats)} combinations of lines")

    quadrilaterals = [
        Polygon.detect(combo, lines, frame, tolerance=4000) for combo in combinats
    ]

    counts = {}
    for quad in quadrilaterals:
        if len(quad.codes) == 0:
            counts.setdefault("VALID", 0)
            counts["VALID"] += 1
        for code in quad.codes:
            counts.setdefault(code, 0)
            counts[code] += 1
    logger.info(f"Counts: {counts}")

    return quadrilaterals, blurred, edges, lines


def find_best_quad(quadrilaterals):
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

    logger.info(f"*** {len(valid_quadris)} VALID QUADRILATERAL FOUND\n\n")

    counts = {}
    for quad in valid_quadris:
        if len(quad.codes) == 0:
            counts.setdefault("VALID", 0)
            counts["VALID"] += 1
        for code in quad.codes:
            counts.setdefault(code, 0)
            counts[code] += 1
    logger.info(counts)

    for quad in valid_quadris:
        logger.info(quad)

    best_quad = None
    if valid_quadris:
        best_quad = min(valid_quadris, key=lambda p: p.color[1])
        logger.info("+++++++++ BEST QUAD ++++++++++")
        logger.info(best_quad)
    return best_quad


def merge_lines(lines, mse_threshold=1):
    """Merges lines based on MSE after regression."""
    merged_lines = []

    for line in lines:
        if not merged_lines:
            merged_lines.append(line)
            continue

        merged = False
        for i in range(len(merged_lines)):
            mse = math.calculate_mse(merged_lines[i], line)
            if mse < mse_threshold:
                x1, y1, x2, y2 = merged_lines[i][0]
                x3, y3, x4, y4 = line[0]
                new_line = [[min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)]]
                merged_lines[i] = new_line
                merged = True
                break

        if not merged:
            merged_lines.append(line)

    lines = [
        [int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])]
        for line in lines
    ]
    merged_lines = [
        [int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])]
        for line in merged_lines
    ]
    return merged_lines


def are_same_quadrilateral(quad1, quad2, tolerance=1.0):
    """
    Check if two quadrilaterals are the same within a given tolerance.
    Allows for cyclic permutations of points.
    """
    # Convert to numpy arrays for easy manipulation

    quad1 = np.array(quad1)
    quad2 = np.array(quad2)

    # Try all cyclic permutations of quad2
    for i in range(len(quad2)):
        permuted_quad2 = np.roll(quad2, shift=i, axis=0)
        if np.allclose(quad1, permuted_quad2, atol=tolerance):
            return True

    return False
