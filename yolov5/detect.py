# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
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
import os
import platform
import sys
from pathlib import Path

import torch

# 获取当前模块的绝对路径，__file__是一个特殊变量，它表示当前模块的文件名
FILE = Path(__file__).resolve()  # 这个文件绝对路径
# 获取了当前模块所在的目录的父目录，即YOLOv5的根目录
ROOT = FILE.parents[0]
# 判断YOLOv5的根目录是否已经在Python模块搜索路径中
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将YOLOv5的根目录添加到Python模块搜索路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 将YOLOv5的根目录转换为相对路径，相对于当前工作目录

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型路径或Triton URL
        source=ROOT / 'data/images',  # 输入源，可以是文件/目录/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # 数据集的yaml文件路径
        imgsz=(640, 640),  # 推断的图像大小（高，宽）
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # 非极大值抑制（NMS）的IOU阈值
        max_det=1000,  # 每个图像的最大检测数
        device='',  # cuda设备，即0或0,1,2,3或cpu
        view_img=False,  # 是否显示检测结果
        save_txt=False,  # 是否保存检测结果到*.txt文件
        save_conf=False,  # 是否在--save-txt标签中保存置信度
        save_crop=False,  # 是否保存裁剪的预测框
        nosave=False,  # 是否不保存图像/视频
        classes=None,  # 按类过滤，例如--class 0，或--class 0 2 3
        agnostic_nms=False,  # 是否为类不可知的NMS
        augment=False,  # 是否进行增强推断
        visualize=False,  # 是否可视化特征
        update=False,  # 是否更新所有模型
        project=ROOT / 'runs/detect',  # 结果保存到的路径
        name='exp',  # 结果保存到的名称
        exist_ok=False,  # 是否允许存在的project/name，不进行递增
        line_thickness=3,  # 边框厚度（像素）
        hide_labels=False,  # 是否隐藏标签
        hide_conf=False,  # 是否隐藏置信度
        half=False,  # 是否使用FP16半精度推断
        dnn=False,  # 是否使用OpenCV DNN进行ONNX推断
        vid_stride=1,  # 视频帧速率步幅
):
    # 将输入源source转换为字符串类型
    source = str(source)
    # 检查输入源是一个txt
    save_img = not nosave and not source.endswith('.txt')
    # 检查输入源是一个文件
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 检查输入源是一个URL
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 检查输入源是否是流文件
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    # 检查输入源是否是摄像头
    screenshot = source.lower().startswith('screen')
    # 是文件或者URL，则检查
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 新建文件夹，/ 操作符用来拼接路径
    # 路径是根据 save_txt 变量的值确定的。如果 save_txt 为 True，则创建的目录是 save_dir 的子目录 labels，否则创建的目录就是 save_dir
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    # 判断后端框架
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
        # 加载图片，将图片抓取
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 推理
    # 随便一张图片热身，以便后续减少延迟
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # dt为时间长度
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # 转为pytorch格式的数组
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0 归一化
            if len(im.shape) == 3:
                im = im[None]  # 扩展一下batch维度，在推理时，模型需要以批处理的形式处理输入数据，因此需要在第一维添加一个维度来表示批大小，x = x[None, :]  # 这里：可以省略，默认在前面扩展

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # 是否输出可视化模型中间特征图（feature map）
            pred = model(im, augment=augment, visualize=visualize)  # augment是否做数据增强

        # NMS
        with dt[2]:
            # 过滤重复部分
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
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 获取图像的尺寸信息，并将其赋值给gn变量，用于之后的归一化处理
            imc = im0.copy() if save_crop else im0  # 是否将预测框裁剪下来保存
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 使用Annotator类对图像进行可视化处理，并传入相关参数，例如图像、线条宽度和名称等
            # 如果有框就把框画出来
            if len(det):
                # 由于放缩到640*640，需要将预测框坐标映射会原图
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 统计每个类别预测框
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 保存结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # 将框信息保存到txt
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # 是否保存截下来的框
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            # 是否将图片显示一下
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # 保存预测结果图片
            if save_img:
                if dataset.mode == 'image':
                    # 保存图片
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    # 保存视频或视频流
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

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 打印结果
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    # 创建了一个 ArgumentParser 类的实例对象，定义 Python 脚本可以接受的命令行参数的方式，这些参数可以包括位置参数和可选参数等
    parser = argparse.ArgumentParser()
    # 接受一个或多个参数值作为模型文件的路径或 Triton 的 URL
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    # 接受的类型
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    # 可选的训练集
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # 图片大小，可以多个参数，长，宽
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # 置信度阈值
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # 做nms的iou阈值,nms算法：每轮选取置信度最大的 Bounding Box（简称 BBox） 接着关注所有剩下的 BBox 中与选取的 BBox 有着高重叠（IoU）的
    # 设置iou阈值理解为预测框和真实框的交并比，适当取该值，淘汰多个框选交叠面积较大的框，选择最大置信度输出
    # 0表明不允许任何交叠，1表明允许全部交叠，中间各值表明大于重叠面积比例淘汰
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # 每个图像的最大检测数量
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # cuda数量
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否展示结果，action='store_true'表示当用户指定了该选项时，将会把该选项的值存储为True
    parser.add_argument('--view-img', action='store_true', help='show results')
    # 是否保存结果到txt
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 是否将置信度保存到txt，单独不报错无效果，需要与save-txt一起加上
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 是否将模型预测出来的目标框对应的图像区域（也就是目标的裁剪图像）保存到本地磁盘上
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # 是否不保存图片或视频
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # --classes选项和目标类别的数字编号来指定需要保留哪些类别的目标。可以多个目标
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 是否使用class-agnostic NMS（非类别感知的非极大值抑制），即在进行非极大值抑制时，忽略目标的类别信息。如果指定了该选项，目标的类别信息将不会用于抑制同类别目标之间的重叠区域。如果未指定该选项，则在进行非极大值抑制时，同类别目标之间的重叠区域将被抑制。
    # 跨类别nms，比如待检测图像中有一个长得很像排球的足球，pt文件的分类中有足球和排球两种，那在识别时这个足球可能会被同时框上2个框：一个是足球，一个是排球。开启agnostic-nms后，那只会框出一个框
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 表示是否进行数据增强后再进行推理。如果指定了该选项，模型会在测试图像上进行多次增强操作，以获得更好的检测效果。如果未指定该选项，则模型将直接在测试图像上进行推理，不进行数据增强
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 表示是否可视化模型中间特征图（feature map）。如果指定了该选项，模型将在推理过程中，将中间特征图可视化输出，以便于分析模型的特征提取能力。如果未指定该选项，则模型不会输出中间特征图。
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # 表示是否更新所有模型。如果指定了该选项，模型将下载并更新最新的所有模型文件。如果未指定该选项，则模型不会更新模型文件，而是使用已经下载好的模型文件进行推理。
    # 指定这个参数，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息
    parser.add_argument('--update', action='store_true', help='update all models')
    # 用于指定保存检测结果的项目名称和目录，其默认值为ROOT / 'runs/detect'，其中ROOT表示YOLOv5根目录。
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 用于指定保存检测结果的名称，默认值为exp。
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 表示如果已经存在同名的项目和名称，是否覆盖原有的检测结果。如果指定了该选项，则表示不增加计数器，直接保存到原有的项目和名称中；如果未指定该选项，则表示在原有的项目和名称上增加计数器，以避免覆盖原有的检测结果。
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 表示边框线的宽度（以像素为单位），默认值为3。在可视化检测结果时，会使用指定宽度的线条来画出目标的边框。
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # 表示是否隐藏检测结果中的标签。如果指定了该选项，则在可视化检测结果时，不显示目标的标签信息。
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # 用于控制是否隐藏目标检测结果中的置信度信息，其默认值为False，表示显示置信度信息。如果指定了该选项，则在可视化检测结果时，不显示目标的置信度信息。
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # 用于指定是否使用FP16半精度浮点数进行推断。如果指定了该选项，则表示使用FP16半精度浮点数进行推断；否则，使用FP32单精度浮点数进行推断。使用FP16可以加速推断过程，但可能会影响检测精度。
    # 低精度技术 (high speed reduced precision)。在training阶段，梯度的更新往往是很微小的，需要相对较高的精度，一般要用到FP32以上。
    # 在inference的时候，精度要求没有那么高，一般F16（半精度）就可以，甚至可以用INT8（8位整型），精度影响不会很大。同时低精度的模型占用空间更小了，有利于部署在嵌入式模型里面
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 用于指定是否使用OpenCV DNN进行ONNX推断。如果指定了该选项，则使用OpenCV DNN进行推断；否则，使用PyTorch进行推断。使用OpenCV DNN可能会加速推断过程，但可能会影响检测精度。
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # 用于控制视频帧的采样率，其默认值为1。如果指定了该选项，表示每隔vid-stride帧进行一次检测。如果未指定该选项，表示对每一帧都进行检测。可以通过指定该选项来加速视频检测的过程。
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # 用于解析命令行参数，并返回一个包含解析结果的Namespace对象。
    option = parser.parse_args()
    # 长度为1的数组则扩大为2数组，如果长度为2则不变，如[640]->[640,640]
    option.imgsz *= 2 if len(option.imgsz) == 1 else 1  # expand
    print_args(vars(option))
    return option


def main(option):
    # 检查运行Yolov5所需要的Python依赖库是否已经安装，但会忽略TensorBoard和THOP这两个库的检查
    check_requirements(exclude=('tensorboard', 'thop'))
    # **vars(option)会将option这个命名空间对象中的所有变量和值以关键字参数的形式展开
    # 换句话说，如果option包含变量a=1和b=2，那么**vars(option)就相当于传递关键字参数a=1, b=2给run函数
    # run(**vars(option))
    # 调用0，表示有一张显卡
    # run(ROOT / 'yolov5m.pt', ROOT / 'data/images', ROOT / 'data/coco128.yaml', (640, 640), 0.25, 0.45, 1000, '0')
    # 训练后结果
    run(ROOT / 'weights/after_train_fire/best.pt', ROOT / 'data/images', ROOT / 'data/fire-smoke.yaml', (640, 640), 0.25, 0.45, 1000, '0')

if __name__ == "__main__":
    # print(torch.cuda.is_available())  # true 查看GPU是否可用
    # print(torch.cuda.device_count())  # GPU数量， 2
    # print(torch.cuda.current_device())  # 当前GPU的索引， 0
    # print(torch.cuda.get_device_name(0))  # 输出GPU名称
    opt = parse_opt()
    main(opt)
