#!/usr/bin/env python3
"""Step-2 平滑版: 在 step2_opencv_basic.py 基础上加两级平滑 (原文件不动)。

用法与 step2_opencv_basic.py 完全相同:
  python3 step2_opencv_smooth.py                    # 默认摄像头
  python3 step2_opencv_smooth.py --source synth     # 合成色块图
  python3 step2_opencv_smooth.py --color blue       # 换预设颜色

为什么原始检测会跳: 每帧独立检测, 噪点/光照微变让掩码边缘闪烁,
外接框对边缘一个噪点像素都敏感。本文件加两个工程上最常用的对策:

知识点:
  1. 指数滑动平均 (EMA): 输出 = a*新值 + (1-a)*旧值
     a 越小越稳但越迟钝 (alpha=1 就退化回原始检测)
  2. 短暂丢失保持 (hold): 检测丢了几帧内沿用上次结果, 避免框闪没
  3. 原始框 (绿细线) 与 平滑框 (青粗线) 同屏对比, 直观看平滑效果

画面状态: detect=检测命中 / hold N=已丢失 N 帧, 沿用上次位置
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

SMOOTH_ALPHA = 0.25  # EMA 新值权重: 越小越稳, 越大越跟手
HOLD_FRAMES = 10     # 检测短暂丢失时, 沿用上次结果多少帧

WIN_IMAGE, WIN_MASK, WIN_ROI, WIN_TUNE = (
    'smooth image', 'smooth mask', 'smooth roi', 'smooth tuning')


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
    """提取指定 HSV 范围内的最大目标, 返回 (mask, box, center, area)。"""
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


def ema(old, new, alpha=SMOOTH_ALPHA):
    """指数滑动平均: alpha 取新值, 1-alpha 保留旧值。"""
    return alpha * new + (1 - alpha) * old


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

    # ---- 平滑状态: 没有历史时为 None ----
    state = None   # {'center': (cx, cy), 'box': (x, y, w, h), 'area': float}
    lost = 0       # 连续丢失帧数

    print('拖动 smooth tuning 窗口的滑条调阈值; q 或 ESC 退出')
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

        # ---- 更新平滑状态 (本文件相对 step2_opencv_basic 的核心改动) ----
        if box is not None:
            # 检测命中: 新测量按 EMA 融进旧状态
            lost = 0
            if state is None:
                state = {'center': center, 'box': box, 'area': area}
            else:
                state = {
                    'center': (ema(state['center'][0], center[0]),
                               ema(state['center'][1], center[1])),
                    'box': tuple(ema(o, n) for o, n in zip(state['box'], box)),
                    'area': ema(state['area'], area),
                }
            label, color_rgb = 'detect', (0, 255, 0)
        elif state is not None and lost < HOLD_FRAMES:
            # 检测丢失但还在保持期内: 沿用上次平滑结果
            lost += 1
            label, color_rgb = f'hold {lost}', (0, 165, 255)
        else:
            # 保持期也过了: 彻底丢目标
            state = None
            label = None

        roi = None
        if state is not None:
            x, y, w, h = map(int, state['box'])
            cx, cy = state['center']

            # 原始检测框: 绿细线 (看得见的抖动)
            if box is not None:
                rx, ry, rw, rh = box
                cv2.rectangle(bgr, (rx, ry), (rx + rw, ry + rh),
                              (0, 255, 0), 1)
            # 平滑后的框: 青粗线 (稳)
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.drawMarker(bgr, (int(cx), int(cy)), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(bgr, f'{label} area={state["area"]:.0f}px',
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color_rgb, 2)
            # ROI 用平滑后的框裁, 尺寸不会逐帧跳变
            roi = bgr[y:y + h, x:x + w]

        if roi is None:
            roi = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.putText(roi, 'no target', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(WIN_IMAGE, bgr)
        cv2.imshow(WIN_MASK, mask)
        cv2.imshow(WIN_ROI, roi)
        key = cv2.waitKey(1 if cap is not None else 30) & 0xFF
        if key in (ord('q'), 27):  # q 或 ESC
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
