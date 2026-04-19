# 手语魔法对战系统 Sign Language Magic Game

> 用真实手语手势施放魔法，基于 MediaPipe + RandomForest 的实时双手手势识别项目

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 演示视频

<!-- 上传视频后把下面链接换成你的视频链接 -->
> 📺 视频演示链接：**[待补充]**

---

## 项目简介

本项目将真实手语动作与游戏魔法系统结合，通过摄像头实时识别双手手势，触发不同的魔法效果与敌人进行对战。

- **数据采集**：自主用摄像头采集手语动作数据，每类手势 600 个样本
- **模型训练**：基于手部 21 个关键点坐标特征，训练 RandomForestClassifier
- **实时推理**：MediaPipe 检测双手关键点，模型实时识别当前手势并触发对应法术

---

## 支持的手势与法术

| 手势 | 手语含义 | 对应法术 | 效果 |
|------|----------|----------|------|
| 双手握拳大拇指交替摆动 | Celebrate（庆祝） | 庆祝咒 | AOE 伤害 20 点 |
| 双手斜伸掌心向外按动 | Help（帮助） | 护盾术 | 反弹伤害 15 点 |
| 双手指向危险方向 | Danger（危险） | 闪电术 | 高额伤害 30 点 |
| 右手食中指按左手腕脉搏 | Sick（生病） | 治疗术 | 恢复生命 40 点 |

---

## 技术方案

```
摄像头画面
    │
    ▼
MediaPipe HandLandmarker
（双手 21 个关键点检测）
    │
    ▼
特征提取
（坐标归一化 + 双手相对位置 + 运动轨迹）
    │
    ▼
RandomForestClassifier
（scikit-learn，自采数据训练）
    │
    ▼
手势标签 + 置信度
    │
    ▼
魔法效果渲染（OpenCV + PIL）
```

### 使用的技术栈

- **MediaPipe** 0.10.x — 实时手部关键点检测
- **scikit-learn** — RandomForestClassifier 手势分类
- **OpenCV** — 摄像头采集与画面渲染
- **PIL / Pillow** — 中文字体渲染
- **joblib** — 模型序列化

---

## 项目结构

```
SignLanguage/
├── SignLanguage.py                  # 主程序（魔法对战游戏）
├── Sign_Language_Data_collection.py # 数据采集工具
├── trained_models/
│   └── latest_sign_language_model.joblib  # 训练好的模型
├── hand_landmarker.task             # MediaPipe 手部检测模型（首次运行自动下载）
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- 摄像头

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行主程序

```bash
python SignLanguage.py
```

首次运行会自动下载 MediaPipe 手部检测模型文件（约 30MB）。

### 4. 操作说明

| 按键 | 功能 |
|------|------|
| `S` | 开始录制手势（需检测到双手） |
| `E` | 结束录制，触发法术 |
| `M` | 恢复魔力 |
| `R` | 重置连击 / 重置敌人 |
| `D` | 调试模式（实时显示识别结果） |
| `ESC` | 退出 |

---

## 自采数据 & 重新训练

如需采集自己的手势数据：

```bash
python Sign_Language_Data_collection.py
```

每类手势自动采集 600 帧样本，保存后可重新训练模型。

---

## requirements.txt

```
opencv-python
mediapipe>=0.10.0
numpy
joblib
scikit-learn
Pillow
```

---

## AI 技术说明

本项目核心 AI 能力：

1. **MediaPipe HandLandmarker**：Google 开源手部关键点检测模型，实时输出双手各 21 个三维坐标点
2. **RandomForestClassifier**：基于自采手语数据集训练，输入为关键点特征向量，输出手势类别与置信度
3. **智能录制分析器**：多帧置信度加权投票，提升单帧抖动下的识别稳定性

---

## License

MIT
