# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import shutil
from datetime import datetime
import math
import os
import platform
import random
import sys
import threading
import time
from pathlib import Path

import requests
import torch
import yaml
from flask import Flask, jsonify, request

from flask_cors import CORS

app = Flask(__name__)
CORS(app,resources=r'/*')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
task_list = {}
delete_task_list = []
@smart_inference_mode()
class AnalysisConfig:
    def __init__(self, min_object_width, min_object_height, score,
                 obj_type, cooldown_duration, frequency):
        self.min_object_width = min_object_width
        self.min_object_height = min_object_height
        self.score = score
        self.obj_type = obj_type
        self.cooldown_duration = cooldown_duration
        self.frequency = frequency
        if frequency != None:
            self.frequency = 1000 / frequency
        if self.cooldown_duration == None:
            print(self.cooldown_duration)
            self.cooldown_duration = 0
def run(
        session,
        analysis_config,
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    start_time = time.time()
    start_f_time = time.time() * 1000
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    print('before model')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    print('111')
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    img_list = []
    # Dataloader
    bs = 1  # batch_size
    while True:
        if webcam:
            print('222')
            #view_img = check_imshow(warn=True)
            view_img = False
            isValid = True
            while True:
                if threading.current_thread().name in delete_task_list:
                    return -1
                if not isValid:
                    time.sleep(10)
                try:
                    print('before loadStreams')
                    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride,
                                          session=session, task_host=task_host)
                    msg = {}
                    msg['taskId'] = threading.current_thread().name
                    timestamp = time.time()
                    formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                    msg['time'] = formatted_time
                    msg['status'] = 1
                    msg['msg'] = '视频流正常'
                    with app.app_context():
                        print('hi')
                        response = session.post(task_host, json=msg)
                    break
                except Exception:
                    msg = {}
                    msg['taskId'] = threading.current_thread().name
                    timestamp = time.time()
                    formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                    msg['time'] = formatted_time
                    msg['status'] = 2
                    msg['msg'] = '视频流异常'
                    isValid = False
                    with app.app_context():
                        response = session.post(task_host, json=msg)
            #dataset = examin_stream(source, imgsz, stride, pt, vid_stride, session)
            if dataset == -1:
                return
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            print(source)

        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            capturedTime = int(time.time() * 1000)
            if threading.current_thread().name != 'MainThread':
                if threading.current_thread().name in delete_task_list:
                    dataset.release()
                    print('release successfully')
                    return 1
            cur_f_time = time.time() * 1000
            if analysis_config.frequency != None:
                if cur_f_time - start_f_time < analysis_config.frequency:
                    continue
            img = {}
            img['data'] = []
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if analysis_config.obj_type is not None and str(int(cls)) not in analysis_config.obj_type:
                            continue
                        if analysis_config.min_object_width is not None:
                            if abs(int(xyxy[0]) - int(xyxy[2])) < analysis_config.min_object_width:
                                continue
                        if analysis_config.min_object_height is not None:
                            if abs(int(xyxy[1]) - int(xyxy[3])) < analysis_config.min_object_width:
                                continue
                        img['data'].append({ 'objectType': int(cls), 'objectName' : names[int(cls)], 'score' : f'{conf:.4f}','leftTopX':int(xyxy[0]),  'leftTopY': int(xyxy[1]),
                                     'rightBtmX': int(xyxy[2]), 'rightBtmY': int(xyxy[3])})
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
            message = ''
            for i, data in enumerate(img['data']):
                if i == 0:
                    message = '检测到' + names[data['objectType']]
                else:
                    message = message + ', 检测到' +names[data['objectType']]
            if message == '':
                message = '未检测到类别'
            img['msg'] = message
            img['resultCd'] = 0
            img['taskId'] = threading.current_thread().name
            img['capturedTime'] = capturedTime
            img['timestamp'] = int(time.time() * 1000)
            img_list.append(img)
            cur_time = time.time()
            if analysis_config.cooldown_duration != 0:
                print(analysis_config.cooldown_duration)
                if cur_time - start_time > analysis_config.cooldown_duration:
                    with app.app_context():
                        response = session.post(data_host, json=img_list)
                    start_time = time.time()
                    img_list = []
            else:
                with app.app_context():
                    response = session.post(data_host, json=[img])
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'output', help='save results to project/name')
    parser.add_argument('--name', default='task', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def picture_run(
        analysis_config = None,
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    img_list = []
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        img = {}
        img['data'] = []
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if analysis_config.obj_type is not None and str(int(cls)) not in analysis_config.obj_type:
                        continue
                    if analysis_config.min_object_width is not None:
                        if abs(int(xyxy[0]) - int(xyxy[2])) < analysis_config.min_object_width:
                            continue
                    if analysis_config.min_object_height is not None:
                        if abs(int(xyxy[1]) - int(xyxy[3])) < analysis_config.min_object_width:
                            continue
                    img['data'].append({'objectType': int(cls), 'objectName': names[int(cls)], 'score': f'{conf:.4f}',
                                        'leftTopX': int(xyxy[0]), 'leftTopY': int(xyxy[1]),
                                        'rightBtmX': int(xyxy[2]), 'rightBtmY': int(xyxy[3])})
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        message = ''
        for i, data in enumerate(img['data']):
            if i == 0:
                message = '检测到' + names[data['objectType']]
            else:
                message = message + ', 检测到' +names[data['objectType']]
        if message == '':
            message = '未检测到类别'
        img['msg'] = message
        img['resultCd'] = 0
        img_list.append(img)
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return img_list
def clear_folder(folder_path):
    try:
        shutil.rmtree(folder_path)  # 删除整个文件夹
        os.mkdir(folder_path)  # 重新创建空文件夹
        print(f"Folder '{folder_path}' cleared.")
    except Exception as e:
        print(f"Error while clearing folder '{folder_path}': {e}")
def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
def main(opt, session, analysis_config):

    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    return run(session, analysis_config, **vars(opt))

def picture_main(opt, analysis_config):

    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    return picture_run(analysis_config, **vars(opt))

@app.route('/picture', methods = ['POST'])
def picture_detection():
    global algCode
    delete_files_in_folder('images/')
    json_data = request.get_json()
    imageUrl = json_data.get('base64')
    analysis_rule = json_data.get('analysisRule')
    analysis_config = AnalysisConfig(
        min_object_width=analysis_rule.get('minObjectWidth'),
        min_object_height=analysis_rule.get('minObjectHeight'),
        score=analysis_rule.get('score'),
        obj_type=analysis_rule.get('objType'),
        cooldown_duration=analysis_rule.get('cooldownDuration'),
        frequency=analysis_rule.get('frequency')
    )
    if imageUrl != '' or imageUrl is not None:
        print(imageUrl)
        try:
            response = requests.get(imageUrl)
            response.raise_for_status()
        except:
            return jsonify([{'resultCd': 1, 'msg': '输入的url有误', 'data':[]}])
        with open(f'images/{random.randint(0, 10000000)}.jpg', 'wb') as file:
            file.write(response.content)
    if algCode == 'chef':
        algCode = algCode + '_mouse'
    opt.weights = f'final_model/{algCode}.pt'
    if gpu_type != 'cpu':
        opt.device = gpu_id
    if analysis_config.score is not None:
        opt.conf_thres = analysis_config.score
    img = picture_main(opt, analysis_config)

    return jsonify(img)

@app.route('/tasks', methods=['POST'])
def createTask():
    global algCode, cur_task
    if algCode == 'chef':
        algCode = algCode + '_mouse'
    json_data = request.get_json()
    task_id = json_data.get('taskId')
    type = json_data.get('type')
    url = json_data.get('url')
    analysis_rule = json_data.get('analysisRule')
    analysis_config = AnalysisConfig(
        min_object_width=analysis_rule.get('minObjectWidth'),
        min_object_height=analysis_rule.get('minObjectHeight'),
        score=analysis_rule.get('score'),
        obj_type=analysis_rule.get('objType'),
        cooldown_duration=analysis_rule.get('cooldownDuration'),
        frequency=analysis_rule.get('frequency')
    )
    opt.weights = f'final_model/{algCode}.pt'
    opt.source = url
    opt.nosave = True
    if gpu_type != 'cpu':
        opt.device = gpu_id
    if analysis_config.score is not None:
        opt.conf_thres = analysis_config.score
    msg = {}
    task_info = {}
    if task_id in task_list:
        msg['resultCd'] = 1
        msg['msg'] = '任务id重复，请重新输入'
        msg['task_id'] = -1
    else:
        msg['resultCd'] = 0
        msg['msg'] = f'创建任务{task_id}成功'
        msg['task_id'] = task_id
        if task_id in delete_task_list:
            delete_task_list.remove(task_id)
        task_info['taskId'] = task_id
        task_info['taskStatus'] = '启动'
        task_info['url'] = url
        session = requests.Session()
        thread = threading.Thread(target=main, args=(opt, session, analysis_config), name=task_id)
        task_list[task_id] = task_info
        thread.start()
        print('thread start succesfully')
    return jsonify(msg)
@app.route('/tasks/<taskId>', methods=['DELETE'])
def deleteTask(taskId):
    res = {}
    print(task_list)
    if taskId not in task_list:
        res['resultCd'] = 1
        res['msg'] = f'未找到{taskId}'
    else:
        res['resultCd'] = 0
        res['msg'] = f'成功删除{taskId}'
        delete_task_list.append(taskId)
        del task_list[taskId]
    return jsonify(res)
@app.route('/tasks/<taskId>', methods=['GET'])
def inquireTask(taskId):
    res = {}
    if taskId == '-1':
        res['resultCd'] = 0
        res['msg'] = '查询所有任务状态成功'
        res['data'] = []
        for taskId in task_list:
            res['data'].append(task_list[taskId])
        return jsonify(res)
    else:
        if taskId in task_list:
            res['resultCd'] = 0
            res['msg'] = f'查询任务{taskId}成功'
            res['data'] = [task_list[taskId]]
            return jsonify(res)
        else:
            res['resultCd'] = 1
            res['msg'] = f'该任务不存在'
            res['data'] = []
            return jsonify(res)
def read_config_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)
    return config_data
def examin_stream(source, imgsz, stride, pt, vid_stride, session):
    isValid = True
    if threading.current_thread().name in delete_task_list:
        return -1
    while True:
        if not isValid:
            time.sleep(10)
        try:
            print('before loadStreams')
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride, session=session, task_host=task_host)
            msg = {}
            msg['taskId'] = threading.current_thread().name
            timestamp = time.time()
            formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
            msg['time'] = formatted_time
            msg['status'] = 1
            msg['msg'] = '视频流正常'
            with app.app_context():
                print('hi')
                response = session.post(task_host, json=msg)
            break
        except Exception:
            msg = {}
            msg['taskId'] = threading.current_thread().name
            timestamp = time.time()
            formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
            msg['time'] = formatted_time
            msg['status'] = 2
            msg['msg'] = '视频流异常'
            isValid = False
            with app.app_context():
                response = session.post(task_host, json=msg)
    return dataset
if __name__ == '__main__':
    opt = parse_opt()
    file_path = "config.yaml"  # 替换为你的config.yaml文件的路径
    config_data = read_config_file(file_path)
    print(config_data)
    ip = config_data['remoteServerIP']
    port = config_data['remoteServerPort']
    algCode = config_data['runtime']['algCode']
    data_host = config_data['httpPush']['captureDataPushUrl']
    task_host = config_data['httpPush']['taskStatusPushUrl']
    gpu_type = config_data['runtime']['gpu-type']
    gpu_id = config_data['runtime']['gpu-id']
    print(algCode)
    app.run(host=ip, port=port)

