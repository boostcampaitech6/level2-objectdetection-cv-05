# ref: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
import argparse
import os
import glob
import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO
from time import process_time


def main(config):
    # submission path로부터 submissions 리스트 생성
    submissions = sorted(glob.glob(f"{config.submissions_dir}/*.csv"))
    submission_df = [pd.read_csv(file) for file in submissions]

    image_ids = submission_df[0]["image_id"].tolist()

    # ensemble 할 file의 image 정보를 불러오기 위한 json
    coco = COCO(config.annotation)

    prediction_strings = []
    file_names = []

    ## configs for ensemble ##
    # iou threshold => nms, soft_nms, nmw, wbf
    iou_thr = config.iou_thr
    # box threshold for skip => soft_nms, nmw, wbf
    skip_box_thr = 0.0001
    # sigma => soft_nms
    sigma = 0.1
    ##########################

    # 각 image id 별로 submission file에서 box좌표 추출
    for i, image_id in enumerate(image_ids):
        prediction_string = ""
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]

        # 각 submission file 별로 prediction box좌표 불러오기
        for df in submission_df:
            predict_string = df[df["image_id"] == image_id][
                "PredictionString"
            ].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list) == 0 or len(predict_list) == 1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                box[0] = float(box[0]) / image_info["width"]
                box[1] = float(box[1]) / image_info["height"]
                box[2] = float(box[2]) / image_info["width"]
                box[3] = float(box[3]) / image_info["height"]
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # 예측 box가 있다면 이를 ensemble 수행
        if len(boxes_list):
            if config.ensemble == "nms":
                boxes, scores, labels = nms(
                    boxes_list, scores_list, labels_list, iou_thr=iou_thr
                )
            elif config.ensemble == "soft_nms":
                boxes, scores, labels = soft_nms(
                    boxes_list,
                    scores_list,
                    labels_list,
                    iou_thr=iou_thr,
                    sigma=sigma,
                    thresh=skip_box_thr,
                )
            elif config.ensemble == "nmw":
                boxes, scores, labels = non_maximum_weighted(
                    boxes_list,
                    scores_list,
                    labels_list,
                    iou_thr=iou_thr,
                    skip_box_thr=skip_box_thr,
                )
            elif config.ensemble == "wbf":
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list,
                    scores_list,
                    labels_list,
                    iou_thr=iou_thr,
                    skip_box_thr=skip_box_thr,
                )
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += (
                    str(int(label))
                    + " "
                    + str(score)
                    + " "
                    + str(box[0] * image_info["width"])
                    + " "
                    + str(box[1] * image_info["height"])
                    + " "
                    + str(box[2] * image_info["width"])
                    + " "
                    + str(box[3] * image_info["height"])
                    + " "
                )

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    os.makedirs(f"{config.submissions_dir}/ensemble", exist_ok=True)
    submission.to_csv(
        f"{config.submissions_dir}/ensemble/submission_{config.ensemble}_{config.iou_thr}_ensemble.csv",
        index=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "--ensemble", type=str, default="wbf", help="ensemble type (default: wbf)"
    )
    parser.add_argument(
        "--iou_thr",
        type=float,
        default=0.4,
        help="ensemble 시 설정할 iou threshold 이 부분을 바꿔가며 대회 metric에 알맞게 적용해봐요! (default: 0.4)",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default=os.environ.get("SM_CAHNNEL_EVAL", "../../dataset/test.json"),
    )
    parser.add_argument(
        "--submissions_dir",
        type=str,
        default=os.environ.get("SM_CAHNNEL_EVAL", "../../sample_submission"),
    )

    args = parser.parse_args()
    print(args)

    start = process_time()
    main(args)
    end = process_time()
    print(f"{args.ensemble} 수행하여 submission_ensemble.csv 생성 완료 \n실행시간: {end - start} s")
