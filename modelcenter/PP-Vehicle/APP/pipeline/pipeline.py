# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import glob
import cv2
import numpy as np
import math
import paddle
import sys
import copy
from collections import defaultdict
from pipeline.datacollector import DataCollector, Result

# add deploy path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from pipeline.cfg_utils import argsparser, merge_cfg
from pipeline.pipe_utils import PipeTimer
from pipeline.pipe_utils import get_test_images, crop_image_with_det, crop_image_with_mot, parse_mot_res, parse_mot_keypoint
from pipeline.pipe_utils import PushStream

from python.infer import Detector, DetectorPicoDet
from python.keypoint_infer import KeyPointDetector
from python.keypoint_postprocess import translate_to_ori_images
from python.preprocess import decode_image, ShortSizeScale
from python.visualize import visualize_box_mask, visualize_attr, visualize_pose, visualize_action, visualize_vehicleplate

from pptracking.python.mot_sde_infer import SDE_Detector
from pptracking.python.mot.visualize import plot_tracking_dict
from pptracking.python.mot.utils import flow_statistic, update_object_info

from pipeline.ppvehicle.vehicle_plate import PlateRecognizer
from pipeline.ppvehicle.vehicle_attr import VehicleAttr

from pipeline.download import auto_download_model


class Pipeline(object):
    """
    Pipeline

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
    """

    def __init__(self, args, cfg):
        self.multi_camera = False
        reid_cfg = cfg.get('REID', False)
        self.enable_mtmct = reid_cfg['enable'] if reid_cfg else False
        self.is_video = False
        self.output_dir = args.output_dir
        self.vis_result = cfg['visual']
        self.input = self._parse_input(args.image_file, args.video_file)

        self.predictor = PipePredictor(args, cfg, self.is_video)
        if self.is_video:
            self.predictor.set_file_name(self.input)

    def _parse_input(self, image, video_file):

        # parse input as is_video and multi_camera

        if image is not None:

            input = image
            self.is_video = False
            self.multi_camera = False

        elif video_file is not None:
            input = video_file
            self.is_video = True
        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file', 'camera_id', 'image_file', 'image_dir']"
            )

        return input

    def run_multithreads(self):

        out = self.predictor.run(self.input)
        return out

    def run(self):

        out = self.predictor.run(self.input)
        return out


def get_model_dir_with_list(cfg, args):
    activate_list = args.avtivity_list
    """ 
        Auto download inference model if the model_path is a url link. 
        Otherwise it will use the model_path directly.
    """
    for key in cfg.keys():
        if type(cfg[key]) == dict and ((key in activate_list) or
                                       "enable" not in cfg[key].keys()):
            if "model_dir" in cfg[key].keys():
                model_dir = cfg[key]["model_dir"]
                downloaded_model_dir = auto_download_model(model_dir)
                if downloaded_model_dir:
                    model_dir = downloaded_model_dir
                    cfg[key]["model_dir"] = model_dir
                print(key, " model dir: ", model_dir)
            elif key == "VEHICLE_PLATE":
                det_model_dir = cfg[key]["det_model_dir"]
                downloaded_det_model_dir = auto_download_model(det_model_dir)
                if downloaded_det_model_dir:
                    det_model_dir = downloaded_det_model_dir
                    cfg[key]["det_model_dir"] = det_model_dir
                print("det_model_dir model dir: ", det_model_dir)

                rec_model_dir = cfg[key]["rec_model_dir"]
                downloaded_rec_model_dir = auto_download_model(rec_model_dir)
                if downloaded_rec_model_dir:
                    rec_model_dir = downloaded_rec_model_dir
                    cfg[key]["rec_model_dir"] = rec_model_dir
                print("rec_model_dir model dir: ", rec_model_dir)

        if (key == "MOT" and (key in activate_list)) or (key == "VEHICLE_PLATE" and (key in activate_list)) or (key == "VEHICLE_ATTR" and (key in activate_list)):  # for idbased and skeletonbased actions
            model_dir = cfg["MOT"]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                model_dir = downloaded_model_dir
                cfg["MOT"]["model_dir"] = model_dir
            print("mot_model_dir model_dir: ", model_dir)


def get_model_dir(cfg):
    """ 
        Auto download inference model if the model_path is a url link. 
        Otherwise it will use the model_path directly.
    """
    for key in cfg.keys():
        if type(cfg[key]) ==  dict and \
            ("enable" in cfg[key].keys() and cfg[key]['enable']
                or "enable" not in cfg[key].keys()):

            if "model_dir" in cfg[key].keys():
                model_dir = cfg[key]["model_dir"]
                downloaded_model_dir = auto_download_model(model_dir)
                if downloaded_model_dir:
                    model_dir = downloaded_model_dir
                    cfg[key]["model_dir"] = model_dir
                print(key, " model dir: ", model_dir)
            elif key == "VEHICLE_PLATE":
                det_model_dir = cfg[key]["det_model_dir"]
                downloaded_det_model_dir = auto_download_model(det_model_dir)
                if downloaded_det_model_dir:
                    det_model_dir = downloaded_det_model_dir
                    cfg[key]["det_model_dir"] = det_model_dir
                print("det_model_dir model dir: ", det_model_dir)

                rec_model_dir = cfg[key]["rec_model_dir"]
                downloaded_rec_model_dir = auto_download_model(rec_model_dir)
                if downloaded_rec_model_dir:
                    rec_model_dir = downloaded_rec_model_dir
                    cfg[key]["rec_model_dir"] = rec_model_dir
                print("rec_model_dir model dir: ", rec_model_dir)

        elif key == "MOT":  # for idbased and skeletonbased actions
            model_dir = cfg[key]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                model_dir = downloaded_model_dir
                cfg[key]["model_dir"] = model_dir
            print("mot_model_dir model_dir: ", model_dir)


class PipePredictor(object):
    """
    Predictor in single camera
    
    The pipeline for image input: 

        1. Detection
        2. Detection -> Attribute

    The pipeline for video input: 

        1. Tracking
        2. Tracking -> Attribute
        3. Tracking -> KeyPoint -> SkeletonAction Recognition
        4. VideoAction Recognition

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
        is_video (bool): whether the input is video, default as False
        multi_camera (bool): whether to use multi camera in pipeline, 
            default as False
    """

    def __init__(self, args, cfg, is_video=True, multi_camera=False):
        # general module for pphuman and ppvehicle
        activate_list = args.avtivity_list
        self.with_mot = True if 'MOT' in activate_list else False

        if self.with_mot:
            print('Multi-Object Tracking enabled')

        # only for ppvehicle
        self.with_vehicleplate = True if 'VEHICLE_PLATE' in activate_list else False
        if self.with_vehicleplate:
            self.with_mot = True
            print('Vehicle Plate Recognition enabled')

        self.with_vehicle_attr = True if 'VEHICLE_ATTR' in activate_list else False
        if self.with_vehicle_attr:
            self.with_mot = True
            print('Vehicle Attribute Recognition enabled')

        self.modebase = {
            "framebased": False,
            "videobased": False,
            "idbased": False,
            "skeletonbased": False
        }

        self.basemode = {
            "MOT": "idbased",
            "ATTR": "idbased",
            "VIDEO_ACTION": "videobased",
            "SKELETON_ACTION": "skeletonbased",
            "ID_BASED_DETACTION": "idbased",
            "ID_BASED_CLSACTION": "idbased",
            "REID": "idbased",
            "VEHICLE_PLATE": "idbased",
            "VEHICLE_ATTR": "idbased",
        }

        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg

        self.output_dir = args.output_dir
        self.draw_center_traj = args.draw_center_traj
        self.secs_interval = args.secs_interval
        self.do_entrance_counting = args.do_entrance_counting
        self.do_break_in_counting = args.do_break_in_counting
        self.region_type = args.region_type
        self.region_polygon = args.region_polygon
        self.illegal_parking_time = args.illegal_parking_time

        self.warmup_frame = self.cfg['warmup_frame']
        self.pipeline_res = Result()
        self.pipe_timer = PipeTimer()
        self.file_name = None
        self.collector = DataCollector()

        self.pushurl = args.pushurl

        # auto download inference model
        get_model_dir_with_list(self.cfg, args)

        if self.with_mot:
            mot_cfg = self.cfg['MOT']
            model_dir = mot_cfg['model_dir']
            tracker_config = mot_cfg['tracker_config']
            batch_size = mot_cfg['batch_size']
            skip_frame_num = mot_cfg.get('skip_frame_num', -1)
            basemode = self.basemode['MOT']
            self.modebase[basemode] = True
            self.mot_predictor = SDE_Detector(
                model_dir,
                tracker_config,
                args.device,
                args.run_mode,
                batch_size,
                args.trt_min_shape,
                args.trt_max_shape,
                args.trt_opt_shape,
                args.trt_calib_mode,
                args.cpu_threads,
                args.enable_mkldnn,
                skip_frame_num=skip_frame_num,
                draw_center_traj=self.draw_center_traj,
                secs_interval=self.secs_interval,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                region_type=self.region_type,
                region_polygon=self.region_polygon)

        if self.with_vehicleplate:
            vehicleplate_cfg = self.cfg['VEHICLE_PLATE']
            self.vehicleplate_detector = PlateRecognizer(args,
                                                         vehicleplate_cfg)
            basemode = self.basemode['VEHICLE_PLATE']
            self.modebase[basemode] = True

        if self.with_vehicle_attr:
            vehicleattr_cfg = self.cfg['VEHICLE_ATTR']
            basemode = self.basemode['VEHICLE_ATTR']
            self.modebase[basemode] = True
            self.vehicle_attr_predictor = VehicleAttr.init_with_cfg(
                args, vehicleattr_cfg)

        if not is_video:
            det_cfg = self.cfg['DET']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            self.det_predictor = Detector(
                model_dir, args.device, args.run_mode, batch_size,
                args.trt_min_shape, args.trt_max_shape, args.trt_opt_shape,
                args.trt_calib_mode, args.cpu_threads, args.enable_mkldnn)
        else:
            if self.with_vehicleplate:
                vehicleplate_cfg = self.cfg['VEHICLE_PLATE']
                self.vehicleplate_detector = PlateRecognizer(args,
                                                             vehicleplate_cfg)
                basemode = self.basemode['VEHICLE_PLATE']
                self.modebase[basemode] = True

    def set_file_name(self, path):
        if path is not None:
            self.file_name = path
        else:
            # use camera id
            self.file_name = None

    def get_result(self):
        return self.collector.get_res()

    def run(self, input, thread_idx=0):
        if self.is_video:
            out = self.predict_video(input, thread_idx=thread_idx)
            return out
        else:
            out = self.predict_image(input)
            return out

    def predict_image(self, input):
        # det
        # det -> attr
        batch_input = [decode_image(input, {})[0]]
        batch_input[0] = cv2.cvtColor(batch_input[0], cv2.COLOR_BGR2RGB)

        # det output format: class, score, xmin, ymin, xmax, ymax
        det_res = self.det_predictor.predict_image(batch_input, visual=False)
        det_res = self.det_predictor.filter_box(det_res,
                                                self.cfg['crop_thresh'])
        self.pipeline_res.update(det_res, 'det')

        if self.with_vehicle_attr:
            crop_inputs = crop_image_with_det(batch_input, det_res)
            vehicle_attr_res_list = []

            for crop_input in crop_inputs:
                attr_res = self.vehicle_attr_predictor.predict_image(
                    crop_input, visual=False)
                vehicle_attr_res_list.extend(attr_res['output'])

            attr_res = {'output': vehicle_attr_res_list}
            self.pipeline_res.update(attr_res, 'vehicle_attr')

        if self.with_vehicleplate:

            crop_inputs = crop_image_with_det(batch_input, det_res)
            platelicenses = []
            for crop_input in crop_inputs:
                platelicense = self.vehicleplate_detector.get_platelicense(
                    crop_input)
                platelicenses.extend(platelicense['plate'])

            vehicleplate_res = {'vehicleplate': platelicenses}
            self.pipeline_res.update(vehicleplate_res, 'vehicleplate')

        return self.visualize_image(batch_input[0], self.pipeline_res)

    def predict_video(self, video_file, thread_idx=0):
        # mot
        # mot -> attr
        # mot -> pose -> action
        capture = cv2.VideoCapture(video_file)

        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("video fps: %d, frame_count: %d" % (fps, frame_count))

        video_name,suffix = os.path.splitext(self.file_name)
        out_path = video_name + "_output"+suffix
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_id = 0

        entrance, records, center_traj = None, None, None
        if self.draw_center_traj:
            center_traj = [{}]
        id_set = set()
        interval_id_set = set()
        in_id_list = list()
        out_id_list = list()
        prev_center = dict()
        records = list()
        if self.do_entrance_counting or self.do_break_in_counting or self.illegal_parking_time != -1:
            if self.region_type == 'horizontal':
                entrance = [0, height / 2., width, height / 2.]
            elif self.region_type == 'vertical':
                entrance = [width / 2, 0., width / 2, height]
            elif self.region_type == 'custom':
                entrance = []
                assert len(
                    self.region_polygon
                ) % 2 == 0, "region_polygon should be pairs of coords points when do break_in counting."
                assert len(
                    self.region_polygon
                ) > 6, 'region_type is custom, region_polygon should be at least 3 pairs of point coords.'

                for i in range(0, len(self.region_polygon), 2):
                    entrance.append(
                        [self.region_polygon[i], self.region_polygon[i + 1]])
                entrance.append([width, height])
            else:
                raise ValueError("region_type:{} unsupported.".format(
                    self.region_type))

        video_fps = fps

        video_action_imgs = []

        object_in_region_info = {
        }  # store info for vehicle parking in region       
        illegal_parking_dict = None

        while (1):
            if frame_id % 10 == 0:
                print('Thread: {}; frame id: {}'.format(thread_idx, frame_id))

            ret, frame = capture.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_id > self.warmup_frame:
                self.pipe_timer.total_time.start()

            if self.modebase["idbased"] or self.modebase["skeletonbased"]:
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].start()

                mot_skip_frame_num = self.mot_predictor.skip_frame_num
                reuse_det_result = False
                if mot_skip_frame_num > 1 and frame_id > 0 and frame_id % mot_skip_frame_num > 0:
                    reuse_det_result = True
                res = self.mot_predictor.predict_image(
                    [copy.deepcopy(frame_rgb)],
                    visual=False,
                    reuse_det_result=reuse_det_result)

                # mot output format: id, class, score, xmin, ymin, xmax, ymax
                mot_res = parse_mot_res(res)
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].end()
                    self.pipe_timer.track_num += len(mot_res['boxes'])

                if frame_id % 10 == 0:
                    print("Thread: {}; trackid number: {}".format(
                        thread_idx, len(mot_res['boxes'])))

                # flow_statistic only support single class MOT
                boxes, scores, ids = res[0]  # batch size = 1 in MOT
                mot_result = (frame_id + 1, boxes[0], scores[0],
                              ids[0])  # single class
                statistic = flow_statistic(
                    mot_result,
                    self.secs_interval,
                    self.do_entrance_counting,
                    self.do_break_in_counting,
                    self.region_type,
                    video_fps,
                    entrance,
                    id_set,
                    interval_id_set,
                    in_id_list,
                    out_id_list,
                    prev_center,
                    records,
                    ids2names=self.mot_predictor.pred_config.labels)
                records = statistic['records']

                if self.illegal_parking_time != -1:
                    object_in_region_info, illegal_parking_dict = update_object_info(
                        object_in_region_info, mot_result, self.region_type,
                        entrance, video_fps, self.illegal_parking_time)
                    if len(illegal_parking_dict) != 0:
                        # build relationship between id and plate
                        for key, value in illegal_parking_dict.items():
                            plate = self.collector.get_carlp(key)
                            illegal_parking_dict[key]['plate'] = plate

                # nothing detected
                if len(mot_res['boxes']) == 0:
                    frame_id += 1
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.img_num += 1
                        self.pipe_timer.total_time.end()
                    if self.cfg['visual']:
                        _, _, fps = self.pipe_timer.get_total_time()
                        im = self.visualize_video(frame, mot_res, frame_id,
                                                  fps, entrance, records,
                                                  center_traj)  # visualize
                        if len(self.pushurl) > 0:
                            pushstream.pipe.stdin.write(im.tobytes())
                        else:
                            writer.write(im)
                            if self.file_name is None:  # use camera_id
                                cv2.imshow('Paddle-Pipeline', im)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                    continue

                self.pipeline_res.update(mot_res, 'mot')
                crop_input, new_bboxes, ori_bboxes = crop_image_with_mot(
                    frame_rgb, mot_res)

                if self.with_vehicleplate and frame_id % 10 == 0:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicleplate'].start()
                    plate_input, _, _ = crop_image_with_mot(
                        frame_rgb, mot_res, expand=False)
                    platelicense = self.vehicleplate_detector.get_platelicense(
                        plate_input)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicleplate'].end()
                    self.pipeline_res.update(platelicense, 'vehicleplate')
                else:
                    self.pipeline_res.clear('vehicleplate')

                if self.with_vehicle_attr:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_attr'].start()
                    attr_res = self.vehicle_attr_predictor.predict_image(
                        crop_input, visual=False)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_attr'].end()
                    self.pipeline_res.update(attr_res, 'vehicle_attr')

            self.collector.append(frame_id, self.pipeline_res)

            if frame_id > self.warmup_frame:
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
            frame_id += 1

            if self.cfg['visual']:
                _, _, fps = self.pipe_timer.get_total_time()

                im = self.visualize_video(frame, self.pipeline_res,
                                          self.collector, frame_id, fps,
                                          entrance, records, center_traj,
                                          self.illegal_parking_time != -1,
                                          illegal_parking_dict)  # visualize
                if len(self.pushurl) > 0:
                    pushstream.pipe.stdin.write(im.tobytes())
                else:
                    writer.write(im)

        if self.cfg['visual'] and len(self.pushurl) == 0:
            writer.release()

        return out_path

    def visualize_video(self,
                        image,
                        result,
                        collector,
                        frame_id,
                        fps,
                        entrance=None,
                        records=None,
                        center_traj=None,
                        do_illegal_parking_recognition=False,
                        illegal_parking_dict=None):
        mot_res = copy.deepcopy(result.get('mot'))
        if mot_res is not None:
            ids = mot_res['boxes'][:, 0]
            scores = mot_res['boxes'][:, 2]
            boxes = mot_res['boxes'][:, 3:]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
            scores = np.zeros([0])

        # single class, still need to be defaultdict type for ploting
        num_classes = 1
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        online_tlwhs[0] = boxes
        online_scores[0] = scores
        online_ids[0] = ids

        if mot_res is not None:
            image = plot_tracking_dict(
                image,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=self.mot_predictor.pred_config.labels,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                do_illegal_parking_recognition=do_illegal_parking_recognition,
                illegal_parking_dict=illegal_parking_dict,
                entrance=entrance,
                records=records,
                center_traj=center_traj)

        vehicle_attr_res = result.get('vehicle_attr')
        if vehicle_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            vehicle_attr_res = vehicle_attr_res['output']
            image = visualize_attr(image, vehicle_attr_res, boxes)
            image = np.array(image)

        if mot_res is not None:
            vehicleplate = False
            plates = []
            for trackid in mot_res['boxes'][:, 0]:
                plate = collector.get_carlp(trackid)
                if plate != None:
                    vehicleplate = True
                    plates.append(plate)
                else:
                    plates.append("")
            if vehicleplate:
                boxes = mot_res['boxes'][:, 1:]
                image = visualize_vehicleplate(image, plates, boxes)
                image = np.array(image)

        return image

    def visualize_image(self, images, result):

        det_res = result.get('det')
        human_attr_res = result.get('attr')
        vehicle_attr_res = result.get('vehicle_attr')
        vehicleplate_res = result.get('vehicleplate')

        if det_res is not None:
            det_res_i = {}
            boxes_num_i = det_res['boxes_num'][0]
            det_res_i['boxes'] = det_res['boxes'][0:0 + boxes_num_i, :]
            im = visualize_box_mask(
                images,
                det_res_i,
                labels=['target'],
                threshold=self.cfg['crop_thresh'])
            im = np.ascontiguousarray(np.copy(im))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        if vehicle_attr_res is not None:
            vehicle_attr_res_i = vehicle_attr_res['output'][0:0 + boxes_num_i]
            im = visualize_attr(im, vehicle_attr_res_i, det_res_i['boxes'])
        if vehicleplate_res is not None:
            plates = vehicleplate_res['vehicleplate']
            det_res_i['boxes'][:, 4:6] = det_res_i[
                'boxes'][:, 4:6] - det_res_i['boxes'][:, 2:4]
            im = visualize_vehicleplate(im, plates, det_res_i['boxes'])

        return im


def pp_vehicls(input_date, avtivity_list):

    paddle.enable_static()

    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    cfg = merge_cfg(FLAGS)  # use command params to update config

    if isinstance(input_date, str):
        FLAGS.video_file = input_date
    else:
        FLAGS.image_file = input_date
    FLAGS.avtivity_list = avtivity_list

    pipeline = Pipeline(FLAGS, cfg)
    out = pipeline.run_multithreads()

    return out
