#!/usr/bin/env python3
"""Step-2: 纯 OpenCV 基础 —— 不碰 ROS、不碰 Gazebo, 先把视觉算法核心跑通。

图像来源三选一 (默认摄像头):
  python3 step2_opencv_basic.py                    # 摄像头
  python3 step2_opencv_basic.py --source synth     # 程序合成的色块图 (离线保底)
  python3 step2_opencv_basic.py --source a.jpg     # 本地图片
  python3 step2_opencv_basic.py --color blue       # 换预设颜色 (只影响滑条初始值)

不需要 source ROS 环境, 只要装了 opencv-python 和 numpy 就能跑。

知识点 (和后面 step4~7 的检测函数是同款流程, 只是在这里先脱离 ROS 练熟):
  1. 图像就是 numpy 数组, OpenCV 通道顺序是 BGR 不是 RGB
  2. BGR -> HSV: 色相范围 0-179, 红色跨 0 所以要两段
  3. inRange 生成掩码 + 形态学开/闭运算去噪
  4. findContours -> 面积过滤 -> 最大轮廓 -> moments 求质心
  5. boundingRect 外接框 + numpy 切片裁出 ROI (目标本身)
  6. createTrackbar 拖滑条实时调阈值:
     拖动 H_lo 越过 H_hi 就自动按"跨 0"拆两段, 正好覆盖红色的情况
"""

import argparse

import cv2
import numpy as np

# 与项目 color_detector.py 的 COLOR_PRESETS 保持一致 (滑条初始值)
HSV_PRESETS = {
    'red': [(0, 80, 60, 10, 255, 255), (170, 80, 60, 179, 255, 255)],
    'blue': [(100, 80, 60, 130, 255, 255)],
    'green': [(35, 60, 60, 85, 255, 255)],
    'yellow': [(20, 80, 60, 40, 255, 255)],
}
MIN_AREA = 150  # 小于这个面积(像素)的轮廓直接忽略

WIN_IMAGE, WIN_MASK, WIN_ROI, WIN_TUNE = (
    'step2 image', 'step2 mask', 'step2 roi', 'step2 tuning')


def make_synth_frame():
    """画一张合成图: 四个不同颜色的色块, 没摄像头也能离线练习。"""
    img = np.full((480, 640, 3), 245, dtype=np.uint8)  # 浅灰底 (BGR)
    cv2.rectangle(img, (80, 100), (220, 240), (170, 60, 60), -1)    # 蓝
    cv2.circle(img, (420, 160), 70, (60, 170, 60), -1)              # 绿
    cv2.rectangle(img, (150, 300), (340, 420), (60, 60, 210), -1)   # 红
    cv2.ellipse(img, (510, 360), (80, 50), 20, 0, 360, (60, 210, 230), -1)  # 黄
    cv2.putText(img, 'synth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    return img


def build_mask(hsv, lo, hi):
    """按 (lo, hi) 范围生成掩码; H_lo > H_hi 时按跨 0 拆成两段 (红色用得上)。"""
    lo, hi = np.array(lo, np.uint8), np.array(hi, np.uint8)
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    mask = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], np.uint8))
    mask |= cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi)
    return mask


def extract_target(bgr, lo, hi):
    """提取指定 HSV 范围内的最大目标。

    返回 (mask, box, center, area):
      mask   二值掩码 (目标=白)
      box    外接框 (x, y, w, h), 没找到为 None
      center 质心 (cx, cy), 没找到为 None
      area   轮廓面积(像素), 没找到为 0
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, lo, hi)

    # 开运算去掉零星噪点, 闭运算填补内部空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if not contours:
        return mask, None, None, 0

    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m['m00'] == 0:
        return mask, None, None, 0

    box = cv2.boundingRect(largest)
    center = (m['m10'] / m['m00'], m['m01'] / m['m00'])
    return mask, box, center, cv2.contourArea(largest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--color', choices=list(HSV_PRESETS), default='red',
                        help='预设颜色, 只决定滑条初始值')
    parser.add_argument('--source', default='cam',
                        help="cam=摄像头(默认), synth=合成图, 或图片路径")
    args = parser.parse_args()

    # ---- 图像来源 ----
    cap = None
    if args.source == 'cam':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise SystemExit('摄像头打不开, 试试 --source synth 用合成图练习')
    elif args.source != 'synth':
        static = cv2.imread(args.source)
        if static is None:
            raise SystemExit(f'读不了图片: {args.source}')

    # ---- 用预设的第一段范围初始化 6 根滑条 ----
    h_lo, s_lo, v_lo, h_hi, s_hi, v_hi = HSV_PRESETS[args.color][0]
    cv2.namedWindow(WIN_TUNE)
    for name, val, maxv in [('H_lo', h_lo, 179), ('H_hi', h_hi, 179),
                            ('S_lo', s_lo, 255), ('S_hi', s_hi, 255),
                            ('V_lo', v_lo, 255), ('V_hi', v_hi, 255)]:
        cv2.createTrackbar(name, WIN_TUNE, val, maxv, lambda x: None)

    print('拖动 step2 tuning 窗口的滑条调阈值; q 或 ESC 退出')
    while True:
        if cap is not None:
            ok, bgr = cap.read()
            if not ok:
                raise SystemExit('摄像头读帧失败')
        elif args.source == 'synth':
            bgr = make_synth_frame()
        else:
            bgr = static.copy()

        pos = lambda n: cv2.getTrackbarPos(n, WIN_TUNE)
        lo = (pos('H_lo'), pos('S_lo'), pos('V_lo'))
        hi = (pos('H_hi'), pos('S_hi'), pos('V_hi'))

        mask, box, center, area = extract_target(bgr, lo, hi)

        roi = None
        if box is not None:
            x, y, w, h = box
            cx, cy = center
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawMarker(bgr, (int(cx), int(cy)), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(bgr, f'area={area:.0f}px', (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # numpy 切片裁出目标本身 —— "提取"的结果
            roi = bgr[y:y + h, x:x + w]

        # ROI 窗口尺寸随目标变化, 没目标时给一块黑图占位
        if roi is None:
            roi = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.putText(roi, 'no target', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(WIN_IMAGE, bgr)
        cv2.imshow(WIN_MASK, mask)
        cv2.imshow(WIN_ROI, roi)
        # 摄像头要 1ms 快速刷新; 静态图慢一点省 CPU
        key = cv2.waitKey(1 if cap is not None else 30) & 0xFF
        if key in (ord('q'), 27):  # q 或 ESC
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
