# mycobot_vision_tutorial

视觉学习路线**阶段 1（热身）**：颜色分割找质心 + 深度反投影 3D 定位。

对应 [docs/视觉方案.md](../docs/视觉方案.md) 中的"入门推荐①：传统视觉（颜色/轮廓分割找物体质心）"。

## 功能

节点 `color_detector`：

1. 同步订阅仿真 D435 的彩色图 `/camera_head/color/image_raw` 和深度图 `/camera_head/depth/image_rect_raw`
2. HSV 颜色分割 + 形态学去噪 + 轮廓检测，取最大色块质心 `(u, v)`
3. 质心处深度取小窗口中位数 `z`（量程受传感器 far clip 限制为 1.5 m）
4. 针孔内参反投影：`x=(u-cx)·z/fx, y=(v-cy)·z/fy` → 相机光学系 3D 点
5. TF 变换到 `base_link`（源坐标系 `camera_head_depth_optical_frame`）
6. 发布结果（均为 5 Hz，随相机）：

| 话题 | 类型 | 内容 |
|---|---|---|
| `/vision/detected_point` | `geometry_msgs/PointStamped` | 目标在 `base_link` 下的 3D 坐标 |
| `/vision/marker` | `visualization_msgs/Marker` | RViz 红色小球标记 |
| `/vision/debug_image` | `sensor_msgs/Image` | 标注了轮廓、十字、坐标的调试图 |

## 编译

```bash
cd ~/ros2_ws
colcon build --packages-select mycobot_vision_tutorial
source install/setup.bash
```

## 运行

终端 1 —— 启动仿真（带相机）：

```bash
ros2 launch mycobot_gazebo mycobot.gazebo.launch.py \
    load_controllers:=true world_file:=pick_and_place_demo.world \
    use_camera:=true use_robot_state_pub:=true use_sim_time:=true
```

终端 2 —— 启动检测节点（默认找红色）：

```bash
ros2 launch mycobot_vision_tutorial color_detector.launch.py target_color:=red
```

或直接运行并调参：

```bash
ros2 run mycobot_vision_tutorial color_detector --ros-args \
    -p use_sim_time:=true -p target_color:=blue -p min_area:=150
```

## 查看结果

- RViz2 中添加 `PointStamped`（`/vision/detected_point`）、`Marker`（`/vision/marker`，Fixed Frame 设为 `base_link`）
- 调试图：`ros2 run rqt_image_view rqt_image_view` 选 `/vision/debug_image`
- 命令行：`ros2 topic echo /vision/detected_point`

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `target_color` | `red` | 预设：`red`/`blue`/`green`/`yellow`/`custom` |
| `hsv_min` / `hsv_max` | `0,80,60` / `10,255,255` | `custom` 模式下的 HSV 上下界，格式 `h,s,v`（H 范围 0-179） |
| `min_area` | `150` | 色块最小像素面积，过滤噪声 |
| `target_frame` | `base_link` | 输出点的参考坐标系 |
| `optical_frame` | `camera_head_depth_optical_frame` | 反投影源坐标系 |
| `depth_window` | `5` | 深度取中位数的窗口边长（像素） |

## 验证小技巧

- `pick_and_place_demo.world` 中的物体若颜色不合适，可在 Gazebo 里拖入带颜色的简单几何体（盒子/球），放到相机视野内（距相机 < 1.5 m）
- RViz 中把检测结果小球与点云 `/camera_head/depth/color/points` 对照，位置应重合——这也是后续阶段 2（PCL 点云管线）的起点

## 进阶：视觉引导移动（move_to_point，C++ 版）

订阅 `/vision/detected_point`，让机械臂末端移动到检测点上方（默认抬高 3 cm，末端朝下）。
**实现在独立 C++ 包 `mycobot_vision_moveit`**（MoveGroupInterface 模式，同 hello_moveit）。

前提：仿真 + move_group 都在运行（可用 `mycobot_bringup/scripts/mycobot_280_gazebo_and_moveit.sh` 一键启动），且 `color_detector` 正在发布检测点。

```bash
# 终端 3（color_detector 之后）
ros2 launch mycobot_vision_moveit move_to_point.launch.py
# 可调参数：approach_height:=0.05（抬高量） single_shot:=false（持续跟随）
```

节点收到第一个有效检测点后会规划并执行一次移动，日志打印 `motion executed, arm is hovering above the target`。
