python detect.py --weights trained/best.pt --source ${1} --img-size 640 --conf-thres 0.6 --cls-conf-thres 0.6 --iou-thres 0.35 --save_dir ${2}
