import cv2
import os.path as op
import tt
import tt.vid
from tt import polygon
import pandas as pd
import json

TEST_DATA = op.join(op.dirname(op.dirname(op.dirname(tt.__file__))), "tests", "data")


def to_dataframe(res):
    players = res.query("type == 'player'")
    best_quads = res.query("type == 'quad' & codes == '[\"BEST\"]'")
    players = [json.loads(e) for e in list(players.vertices)]
    best_quads = [json.loads(e) for e in list(best_quads.vertices)]
    return players, best_quads


def test_frame():
    fp = op.join(TEST_DATA, "frame.png")
    csv = pd.read_csv(op.join(TEST_DATA, "frame.csv"))

    print(fp)
    frame = cv2.imread(fp)

    net = cv2.dnn.readNet(
        op.join(op.dirname(tt.__file__), "data/yolov4.weights"),
        op.join(op.dirname(tt.__file__), "data/yolov4.cfg"),
    )
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    f, _ = tt.vid.process_frame(frame, net, frame_index=1)
    res = f.to_dataframe(fps=120, start_index=524)
    test_players, test_quads = to_dataframe(res)
    gt_players, gt_quads = to_dataframe(csv)
    assert polygon.are_same_quadrilateral(test_quads, gt_quads)
    assert test_players == gt_players
    return True
