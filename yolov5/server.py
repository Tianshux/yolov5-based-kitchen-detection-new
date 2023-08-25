import argparse
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

from detect import main
from utils.general import print_args

app = Flask(__name__)
CORS(app,resources=r'/*')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
task_list = {}
delete_task_list = []
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
            self.cooldown_duration = 0
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
@app.route('/detect', methods = ['POST'])
def kitchen_detect():
    json_data = request.get_json()
    imgBase = json_data.get('imageBase64')
    imageUrl = json_data.get('imageUrl')
    algCode = json_data.get('algCode')
    if imgBase == None or imgBase == 'None':
        response = requests.get(imageUrl)
        response.raise_for_status()
        with open(f'images/{random.randint(0, 10000000)}.jpg', 'wb') as file:
            file.write(response.content)
    if algCode == 'chef':
        algCode = algCode + '_mouse'
    opt.weights = f'final_model/{algCode}.pt'
    opt.save_txt = True
    img_list = main(None, opt)
    return jsonify(img_list)

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
        task_list[task_id] = thread
        thread.start()
        print('thread start succesfully')
    return jsonify(msg)
@app.route('/tasks/<int:taskId>', methods=['DELETE'])
def deleteTask(taskId):
    res = {}
    if taskId not in task_list:
        res['resultCd'] = 1
        res['msg'] = f'未找到{taskId}'
    else:
        res['resultCd'] = 0
        res['msg'] = f'成功删除{taskId}'
        delete_task_list.append(taskId)
        del task_list['taskId']


def read_config_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)
    return config_data
if __name__ == '__main__':
    opt = parse_opt()
    file_path = "config.yaml"  # 替换为你的config.yaml文件的路径
    config_data = read_config_file(file_path)
    print(config_data)
    ip = config_data['remoteServerIP']
    port = config_data['remoteServerPort']
    algCode = config_data['runtime']['algCode']
    host = config_data['httpPush']['host']
    data_host = host + config_data['httpPush']['captureDataPushUrl']
    task_host = host + config_data['httpPush']['taskStatusPushUrl']
    print(algCode)
    app.run(host=ip, port=port)