import os
import csv
import json
import argparse
import time
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from rect_tracker import RectTracker as Tracker

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    save_json = False
    save_txt = False
    # Directories
    # save_dir = opt.save_dir
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    save_dir = Path(opt.save_dir)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = True
    if classify:
        weightsPath = os.path.join("trained", "efficientnet_b0_1587500.weights")
        configPath = os.path.join("trained", "efficientnet_b0.cfg")
        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        with open(os.path.join("trained", "labels.list")) as f:
            names = f.read().splitlines()
            names.append("Unknown")
        # modelc = load_classifier(name='resnet101', n=2)  # initialize
        # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # colors[0] = [170, 20, 255]
    colors[-1] = [170, 20, 255]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    csv_f = open(os.path.join(opt.save_dir, "detections.csv"), "w")
    csv_writer = csv.writer(csv_f, delimiter=",")

    csv_writer.writerow(['video_name', 'frame_no', 'row1', 'col1', 'row2', 'col2', 'row3', 'col3', 'row4', 'col4', 'fish_specie', 'tracking_id'])

    tracker = Tracker()

    json_out = None

    for path, img, im0s, vid_cap in dataset:
        json_out_frame = []
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            for i, det in enumerate(pred):
                det = torch.cat((det, torch.Tensor(np.zeros((det.shape[0], 1))).to(device)), dim=1)
                pred[i] = det
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    ln = net.getLayerNames()
                    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                    # ln = ['conv_130'] + [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                    # ln = ln[-20:]
                    for bi, box in enumerate(det):
                        box = box.long()
                        # print(box)
                        crop = im0s[box[1]:box[3], box[0]:box[2]]
                        blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (224, 224),
                            swapRB=True, crop=False)
                        net.setInput(blob)
                        layerOutputs = net.forward(ln)
                        c       = layerOutputs[0][0, :, 0, 0].argmax()
                        conf    = layerOutputs[0][0, :, 0, 0].max()
                        if conf > opt.cls_conf_thres:
                            det[bi, -2] = torch.Tensor([c]).to(device)
                            det[bi, -1] = torch.Tensor([conf]).to(device)
                        else:
                            det[bi, -2] = -1
                            det[bi, -1] = -1
                            # det[bi, -1] = torch.Tensor([conf]).to(device)

                        # cv2.imwrite("crops/{}".format(os.path.basename(path) + '_{}_{}'.format(str(dataset.frame), bi) + ".png"), crop)

                    # determine only the *output* layer names that we need from YOLO
                    # construct a blob from the input image and then perform a forward
                    # pass of the YOLO object detector, giving us our bounding boxes and
                    # associated probabilities

            # pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / (p.stem if dataset.mode == 'image' else p.name)) + ('' if dataset.mode == 'image' else '_{:04}'.format(frame - 1))  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if vid_path != save_path:  # new video
                tracker = Tracker()

            if True:
                # Print results
                for c in det[:, -2].unique():
                    n = (det[:, -2] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                tracker.update(det.cpu().numpy())

                # Write results
                if save_txt:
                    with open(txt_path + '.txt', 'w') as f:
                        pass
                for object_id, (x1, y1, x2, y2, d_conf, cls, c_conf) in tracker.objects.items():
                    if tracker.disappeared[object_id] == 0:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        csv_writer.writerow([os.path.basename(path), frame, x1, y1, x2, y1, x2, y2, x1, y2, names[int(cls)], object_id])
                        json_out_frame.append({"bbox": tuple(map(int, [x1, y1, x2 - x1, y2 - y1])), "d_conf": float(d_conf), "class": names[int(cls)], "c_conf": float(c_conf), "tracking_id": int(object_id)})

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor([x1,y1,x2,y2]).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (object_id, *xywh, d_conf, c_conf) if opt.save_conf else (object_id, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{object_id}'# {cls} {c_conf:.2f}'
                            plot_one_box((x1, y1, x2, y2), im0, label=label, color=colors[object_id], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        if save_json and isinstance(json_out, list):
                            with open(vid_path.rsplit('.', 1)[0] + '.json', 'w', encoding='utf-8') as f:
                                json.dump(json_out, f, ensure_ascii=False, indent=4)

                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        json_out = []

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
            
            if dataset.mode == 'video':   
                json_out.append(json_out_frame)

    if save_json and isinstance(json_out, list):
        with open(vid_path.rsplit('.', 1)[0] + '.json', 'w', encoding='utf-8') as f:
            json.dump(json_out, f, ensure_ascii=False, indent=4)
    csv_f.close()
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--cls-conf-thres', type=float, default=0.25, help='class confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_dir', action='store', help='save results to {save_dir}')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
