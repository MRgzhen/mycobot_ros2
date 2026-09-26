# 视觉 Demo 分步教程

从零实现 [color_detector.py](../color_detector.py) 的分步练习，分三个阶段，每一步只新增一个知识点：

1. **纯 OpenCV**（step2）：不碰 ROS 和 Gazebo，先把视觉算法核心练熟
2. **ROS2 基础**（step1）：发布/订阅、定时器、回调
3. **两者结合**（step3~8）：订阅 Gazebo 相机图像，逐步加处理、发布、深度、TF

另外附两个独立小工具：视觉提取（extra，选做）和检测跟踪（step8）。

## 环境

- ROS2 Humble
- 依赖包：`ros-humble-cv-bridge`、`ros-humble-tf2-geometry-msgs`、OpenCV

step1 需要 ROS 环境，每个终端先执行：

```bash
source /opt/ros/humble/setup.bash
```

**step2 是纯 OpenCV 脚本，不需要 source ROS 环境**，只依赖 `opencv-python` 和 `numpy`。

## 步骤总览

| 步骤 | 脚本 | 学什么 | 需要仿真 | 需要 ROS |
|---|---|---|---|---|
| 1a | step1_talker.py | 发布器 + 定时器 | 否 | 是 |
| 1b | step1_listener.py | 订阅器 + 回调 | 否 | 是 |
| 2 | step2_opencv_basic.py | 纯 OpenCV：HSV、掩码、轮廓、质心、ROI、滑条调参 | 否 | **否** |
| 3 | step3_image_listener.py | 图像订阅、QoS、cv_bridge 细节 | 是 | 是 |
| 4 | step4_color_detect.py | HSV 颜色检测 + 质心 | 是 | 是 |
| — | extra_vision_extract.py | 选做：在相机流上裁 ROI（step2+4 综合） | 是 | 是 |
| 5 | step5_publish_debug.py | 发布调试图（完整闭环） | 是 | 是 |
| 6 | step6_depth_3d.py | 时间同步 + 深度 + 反投影 3D | 是 | 是 |
| 7 | step7_tf_marker.py | TF 坐标变换 + Marker | 是 | 是 |
| 8 | step8_detect_track.py | 目标检测 + 跟踪（不发布） | 是 | 是 |

所有脚本直接用 `python3` 运行，不需要 colcon build。

## Step 1a：最小发布器

```bash
cd ~/ros2_ws/src/mycobot_ros2/mycobot_vision_tutorial/mycobot_vision_tutorial/demo
python3 step1_talker.py
```

另开终端验证：

```bash
ros2 topic list                    # 能看到 /demo/chatter
ros2 topic echo /demo/chatter      # 打印消息内容
ros2 topic hz /demo/chatter        # 显示频率 (约 2 Hz)
```

## Step 1b：最小订阅器

保持 1a 运行，另开终端：

```bash
python3 step1_listener.py
```

看到 `heard: hello N` 日志即成功。`Ctrl+C` 退出。

## Step 2：纯 OpenCV 基础（不碰 ROS、不碰 Gazebo）

把后面所有步骤要用的视觉算法核心，先在纯 Python 环境跑通：图像就是 numpy 数组、
BGR→HSV、inRange 掩码、形态学去噪、轮廓、质心、外接框、numpy 切片裁 ROI。

```bash
python3 step2_opencv_basic.py                    # 默认摄像头
python3 step2_opencv_basic.py --source synth     # 没摄像头: 用合成色块图
python3 step2_opencv_basic.py --source a.jpg     # 或用本地图片
python3 step2_opencv_basic.py --color blue       # 换预设颜色(滑条初始值)
```

四个窗口：
- `step2 image`：原图 + 绿色外接框 + 黄色质心十字 + 面积
- `step2 mask`：黑白掩码（目标=白）
- `step2 roi`：裁出来的目标区域本身（没目标时显示 no target）
- `step2 tuning`：6 根 HSV 阈值滑条，拖动实时看效果

对着摄像头拿一个彩色物体，拖 `step2 tuning` 里的滑条直到 mask 里只剩目标——
这一步调阈值的手感，后面所有步骤都用得上。红色跨色相 0 时，把 H_lo 拖到比
H_hi 大即可自动按"跨 0"拆两段。`q` 或 `ESC` 退出。

## 启动仿真（Step 3 起 prerequisite）

```bash
ros2 launch mycobot_bringup mycobot.gazebo.launch.py use_camera:=true
```

## Step 3：订阅相机图像

```bash
python3 step3_image_listener.py
```

弹出窗口显示相机画面。没画面先检查话题：

```bash
ros2 topic list | grep camera_head
ros2 topic hz /camera_head/color/image_raw
```

## Step 4：颜色检测（HSV + 轮廓 + 质心）

```bash
python3 step4_color_detect.py
```

显示两个窗口：`step4 color`（画面 + 黄色十字质心）、`step4 mask`（黑白掩码）。
默认检测红色；在场景里放一个红色方块（Gazebo 里可以手动拖入模型改颜色）。
要换颜色，改脚本里的 `HSV_RANGES['red']` 为 `'blue'` 等预设。
阈值不准时，回到 step2 用滑条调出合适的值再填进来。

## 选做：视觉提取 extra_vision_extract.py

step2 + step4 的综合练习：在实时相机流上把目标"抠"出来（不发布任何话题）。

```bash
python3 extra_vision_extract.py
python3 extra_vision_extract.py --color blue
```

窗口与 step2 相同（image / mask / roi），只是图像来源换成了 Gazebo 相机话题，
多练一遍 cv_bridge 和传感器 QoS。

## Step 5：发布调试图（订阅 → 处理 → 发布闭环）

```bash
python3 step5_publish_debug.py
```

另开终端用 rqt 查看：

```bash
ros2 run rqt_image_view rqt_image_view
```

下拉框选 `/vision/debug_image`，看到绿色轮廓 + 黄色十字标注即成功。

## Step 6：深度 + 3D 坐标

```bash
python3 step6_depth_3d.py
```

终端周期性输出：

```
px=(320,240) -> optical: x=0.012 y=-0.005 z=0.450 m
```

前两步日志先打印相机内参 `intrinsics: fx=... fy=... cx=... cy=...`。

## Step 7：TF 变换 + Marker

```bash
python3 step7_tf_marker.py
```

另开终端启动 RViz：

```bash
rviz2
```

配置：
1. Fixed Frame 选 `base_link`
2. Add → By topic → `/vision/marker` → Marker
3. （可选）Add → By topic → `/vision/detected_point` → PointStamped

看到机器人基座前方一个红色小球，即完整复现原版功能。

## Step 8：检测目标 + 跟踪目标（不发布任何话题）

检测与跟踪分工的独立小工具（也只用 imshow，没有 publisher）：

- **检测**：每帧 HSV 分割找目标（和 step2 同款流程）
- **跟踪**：CSRT 跟踪器记住目标外观，检测短暂丢失（遮挡、光照突变）时顶上

```bash
python3 step8_detect_track.py
python3 step8_detect_track.py --color blue    # 换颜色
```

窗口里看状态：绿框 `detect`（检测命中）、蓝框 `track`（检测丢失、跟踪器接管）、红字 `searching...`（全丢，重新搜索）。紫色折线是运动轨迹，框上方显示实时速度 px/s。

在 Gazebo 里拖动色块，能看到轨迹线跟着画出来；用手/遮挡物挡住色块几帧再移开，观察 detect → track → detect 的切换。

## 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| step1 listener 收不到消息 | talker 没运行，或两边没 source 同一个 ROS 环境 |
| step2 摄像头打不开 | 设备被占用或不存在，用 `--source synth` 离线练习 |
| step2 mask 全黑/全白 | 阈值不对，拖 tuning 滑条；确认物体颜色饱和度够 |
| step3 起黑屏/无窗口 | 仿真没开 `use_camera:=true`；或 QoS 不匹配（脚本已配 BEST_EFFORT） |
| step4/extra/8 找不到色块 | 阈值不对，先回 step2 用滑条调出值再改脚本；或物体面积小于 150 px |
| step6 报 no valid depth | 物体超出 `DEPTH_MAX=1.5` 米或深度无效，把物体放近一点 |
| step7 报 TF failed | 仿真模型没发布 TF，等几秒；或 frame 名不对，用 `ros2 run tf2_tools view_frames` 查 |
| step8 卡顿 | CSRT 精度高但慢，把 `TrackerCSRT_create` 换成 `TrackerKCF_create` 可提速 |
| cv_bridge 导入失败 | `sudo apt install ros-humble-cv-bridge` |

## 与原版的对应关系

| 原版模块 | demo 对应 |
|---|---|
| detect_blob() | step2 纯 OpenCV 先练 → step4/5/6/7 的检测函数 |
| HSV 阈值调参 | step2 的 tuning 滑条（原版是写死的 COLOR_PRESETS） |
| ROI 裁剪 (numpy 切片) | step2 / extra |
| read_depth() | step6/7 的 read_depth() |
| 针孔反投影 | step6/7 的 on_images() |
| TF + Marker | step7 |
| 参数声明 COLOR_PRESETS / declare_parameter | demo 写死为常量，理解流程后再看参数化写法 |
| 无（扩展练习） | extra 视觉提取、step8 检测跟踪 —— 原版没有对应模块，是独立小工具 |
