# ${변수} 정의
ENSEMBLE='nmw'
THR=0.5
ANN="/data/ephemeral/home/level2-objectdetection-cv-05/data/dataset/test.json"
SUB_DIR="/data/ephemeral/home/level2-objectdetection-cv-05/code/submissions"

# run with args
python tools/ensemble.py \
--ensemble ${ENSEMBLE} \
--iou_thr ${THR} \
--annotation ${ANN} \
--submissions_dir ${SUB_DIR}