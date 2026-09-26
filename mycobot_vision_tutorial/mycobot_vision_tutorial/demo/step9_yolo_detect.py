#!/usr/bin/env python3
"""Step-9: YOLO 目标检测入门 —— 纯离线, 不碰 ROS, 不碰 Gazebo (YOLO 阶段 1)。

前置: pip install ultralytics
      (首次运行会自动下载 yolo11n.pt 权重约 5MB, 需要网络)
运行: python3 step9_yolo_detect.py                  # 摄像头
      python3 step9_yolo_detect.py a.jpg            # 本地图片
      python3 step9_yolo_detect.py --conf 0.3       # 降低置信度门槛
      python3 step9_yolo_detect.py --model yolo11s  # 换更大的模型

和 step2 一样只需要 Python 环境, 不用 source ROS。

知识点:
  1. YOLO 用法三步: YOLO('xxx.pt') 加载一次 -> model(frame) 推理 -> 解析 boxes
     model(frame) 这一行的角色 = step4 的 detect_blob(), 只是判据从
     "HSV 颜色"换成了"深度学习特征", 后面管线(取中心->查深度->反投影)完全复用
  2. 每个输出框的三个量: xyxy 坐标 / conf 置信度 / cls 类别号
  3. conf 置信度阈值: 角色 = step2 的 MIN_AREA, 都是"把弱证据丢掉"
  4. 模型规格 yolo11n < yolo11s < yolo11m: 越大越准也越慢, n 在 CPU 上就能跑
  5. 框中心的黄十字 = 将来接入 step6 深度反投影的接口点 (拆码垛抓取点)

注意: 预训练模型只认 COCO 80 类 (person/cup/bottle/book...), 没有"纸箱"。
纸箱要等阶段 2: 自己标几百张图微调。本步先把推理管线跑通、建立手感。
"""

import argparse
import time
from collections import Counter

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        '缺 ultralytics 库, 先安装: pip install ultralytics '
        '(首次运行还需联网下载权重)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11n',
                        help='yolo11n(最快)/yolo11s/yolo11m, 越大越准越慢')
    parser.add_argument('--conf', type=float, default=0.4,
                        help='置信度门槛, 低于它的框直接丢弃')
    parser.add_argument('source', nargs='?', default='cam',
                        help='cam=摄像头(默认), 或图片路径')
    args = parser.parse_args()

    # ---- 模型: 加载一次, 循环里反复推理 ----
    model = YOLO(args.model + '.pt')

    # ---- 图像来源 ----
    cap = None
    if args.source == 'cam':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise SystemExit('摄像头打不开, 换成图片路径试试')
    else:
        static = cv2.imread(args.source)
        if static is None:
            raise SystemExit(f'读不了图片: {args.source}')

    names = model.names  # {0: 'person', 1: 'bicycle', ...} COCO 80 类
    fps = 0.0            # EMA 平滑的推理帧率 (第一帧预热慢, 属正常)

    print('q 或 ESC 退出')
    while True:
        if cap is not None:
            ok, bgr = cap.read()
            if not ok:
                raise SystemExit('摄像头读帧失败')
        else:
            bgr = static.copy()

        # ---- 推理: 就这一行 (verbose=False 关掉库自带的逐帧刷屏日志) ----
        t0 = time.perf_counter()
        results = model(bgr, conf=args.conf, verbose=False)
        dt = time.perf_counter() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

        # ---- 解析输出: 和 step2 手动画框完全同一套 cv2 画图代码 ----
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f'{names[int(box.cls[0])]} {float(box.conf[0]):.2f}'
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bgr, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 框中心: 将来拿这个 (cx, cy) 去深度图查 Z, 再反投影成 3D
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(bgr, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 15, 2)

        # 终端摘要: 每类目标各检出几个, 如 "person x2, cup x1"
        counts = Counter(names[int(b.cls[0])] for b in boxes)
        summary = ', '.join(f'{k} x{v}' for k, v in counts.items()) or 'none'
        cv2.putText(bgr, f'{len(boxes)} objs, {fps:.1f} FPS',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow('step9 yolo', bgr)
        if len(boxes):
            print(f'detected: {summary}  ({fps:.1f} FPS)', end='\r')
        key = cv2.waitKey(1 if cap is not None else 30) & 0xFF
        if key in (ord('q'), 27):  # q 或 ESC
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
