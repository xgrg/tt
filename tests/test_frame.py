import cv2
import os.path as op
import tt
import tt.vid
import pandas as pd
from loguru import logger

TEST_DATA = op.join(op.dirname(op.dirname(op.dirname(tt.__file__))), "tests", "data")


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

    f, _ = tt.vid.process_frame(frame, net, frame_index=0)
    res = f.to_dataframe(fps=25, start_index=524)
    logger.info(res)
    logger.info(csv)
