# complete_sign_language_trainer.pimport cv2
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import joblib
from datetime import datetime
import sys
import time
import urllib.request
from PIL import Image, ImageDraw, ImageFont

# ======================================================
# MediaPipe 兼容层（支持 mediapipe >= 0.10.x）
# ======================================================
def _ensure_hand_model(path="hand_landmarker.task"):
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print(f"正在下载手部检测模型到 {path} ...")
        try:
            urllib.request.urlretrieve(url, path)
            print("模型下载完成")
        except Exception as e:
            raise RuntimeError(f"模型下载失败: {e}\n请手动下载:\n{url}")
    return path

class _FakeLandmark:
    def __init__(self, lm):
        self.x = lm.x; self.y = lm.y; self.z = lm.z

class _FakeHandLandmarks:
    HAND_CONNECTIONS = frozenset([
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ])
    def __init__(self, landmarks):
        self.landmark = [_FakeLandmark(lm) for lm in landmarks]

class _FakeHandedness:
    class _Cat:
        def __init__(self, label, score):
            self.label = label; self.score = score
    def __init__(self, category):
        self.classification = [self._Cat(category.category_name, category.score)]

class _FakeResults:
    def __init__(self, lms, heds):
        self.multi_hand_landmarks = lms if lms else None
        self.multi_handedness     = heds if heds else None

class _NewAPIHandsWrapper:
    HAND_CONNECTIONS = _FakeHandLandmarks.HAND_CONNECTIONS
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.3,
                 model_complexity=1):
        from mediapipe.tasks import python as _mpp
        from mediapipe.tasks.python import vision as _vis
        model_path = _ensure_hand_model("hand_landmarker.task")
        mode = _vis.RunningMode.IMAGE if static_image_mode else _vis.RunningMode.VIDEO
        opts = _vis.HandLandmarkerOptions(
            base_options=_mpp.BaseOptions(model_asset_path=model_path),
            running_mode=mode, num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self._lm = _vis.HandLandmarker.create_from_options(opts)
        self._mode = mode
        self._ts = 0
        from mediapipe.tasks.python.vision import RunningMode as _RM
        self._RM = _RM
    def process(self, rgb_frame):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self._mode == self._RM.VIDEO:
            self._ts += 33
            raw = self._lm.detect_for_video(img, self._ts)
        else:
            raw = self._lm.detect(img)
        lms  = [_FakeHandLandmarks(h) for h in raw.hand_landmarks]
        heds = [_FakeHandedness(raw.handedness[i][0]) for i in range(len(raw.handedness))]
        return _FakeResults(lms, heds)
    def __enter__(self): return self
    def __exit__(self, *a): self._lm.close()

class _DrawingUtilsCompat:
    class DrawingSpec:
        def __init__(self, color=(255,255,255), thickness=2, circle_radius=2):
            self.color = color; self.thickness = thickness; self.circle_radius = circle_radius
    def draw_landmarks(self, image, hand_landmarks, connections,
                       landmark_drawing_spec=None, connection_drawing_spec=None):
        h, w = image.shape[:2]
        lms = hand_landmarks.landmark
        cs = connection_drawing_spec or self.DrawingSpec(color=(255,0,0), thickness=2)
        for s, e in (connections or []):
            cv2.line(image, (int(lms[s].x*w), int(lms[s].y*h)),
                             (int(lms[e].x*w), int(lms[e].y*h)), cs.color, cs.thickness)
        ls = landmark_drawing_spec or self.DrawingSpec(color=(0,255,0), circle_radius=2)
        for lm in lms:
            cv2.circle(image, (int(lm.x*w), int(lm.y*h)), ls.circle_radius, ls.color, ls.thickness)

# 替代 mp.solutions.hands / mp.solutions.drawing_utils
class _SolutionsShim:
    class _HandsFactory:
        HAND_CONNECTIONS = _FakeHandLandmarks.HAND_CONNECTIONS
        def Hands(self, **kwargs):
            return _NewAPIHandsWrapper(**kwargs)
    hands = _HandsFactory()
    drawing_utils = _DrawingUtilsCompat()

# 注入到 mp 命名空间，让旧代码 mp.solutions.xxx 正常工作
if not hasattr(mp, 'solutions'):
    mp.solutions = _SolutionsShim()
# ======================================================


class ChineseDisplaySupport:
    """中文显示支持类"""
    
    @staticmethod
    def get_chinese_font(font_size=20):
        """获取中文字体"""
        import platform
        system = platform.system()
        
        font_paths = []
        
        if system == "Windows":
            # Windows常见中文字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",      # 黑体
                "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
                "C:/Windows/Fonts/simsun.ttc",      # 宋体
                "C:/Windows/Fonts/simkai.ttf",      # 楷体
            ]
        elif system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
            ]
        elif system == "Linux":
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
            ]
        
        # 尝试加载字体
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except:
                    continue
        
        # 如果找不到中文字体，返回默认字体
        try:
            return ImageFont.truetype(None, font_size)
        except:
            return ImageFont.load_default()
    
    @staticmethod
    def put_chinese_text(image, text, position, font_size=20, color=(255, 255, 255)):
        """
        在图像上绘制中文文本
        
        Args:
            image: OpenCV图像 (BGR格式)
            text: 要绘制的文本
            position: 位置 (x, y)
            font_size: 字体大小
            color: 颜色 (B, G, R)
            
        Returns:
            image: 绘制文本后的图像
        """
        # 将OpenCV图像转换为PIL图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 获取字体
        font = ChineseDisplaySupport.get_chinese_font(font_size)
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color[::-1])  # PIL使用RGB，OpenCV使用BGR
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

class CompleteSignLanguageCollector:
    def __init__(self, data_dir="complete_sign_data"):
        """
        完整的真实手语数据收集器（支持中文显示）
        
        Args:
            data_dir: 数据保存目录
        """
        self.data_dir = data_dir
        self.landmark_data = []
        self.labels = []
        self.current_label = None
        self.collecting = False
        self.counter = 0
        self.max_samples_per_gesture = 600  # 每个手势600个样本
        self.frame_count = 0
        self.prev_landmarks = None
        
        # 创建数据目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 初始化MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 完整的手语定义（基于你的建议）
        self.complete_sign_language = {
            "celebrate": {
                "chinese": "庆祝术",
                "english": "Celebrate",
                "sign_description": "首先，右手竖起大拇指，左手平放，右手划过左手顶部，接着，左手握拳，右手食指划过左手四指顶部，最后，双手掌心向上交替摆动（新年快乐）",
                "action_steps": [
                    "双手握拳，大拇指竖起",
                    "交替上下摆动大拇指",
                    "面带微笑，轻松自然",
                    "摆动3-5次完成动作"
                ],
                "magic_meaning": "🎉 释放庆祝魔法，带来欢乐和祝福",
                "sample_count": 0
            },
            "help": {
                "chinese": "护盾术",
                "english": "Help",
                "sign_description": "双手斜伸，掌心向外，按动两下（标准'帮助'手语）",
                "action_steps": [
                    "双手向前斜伸45度",
                    "掌心朝外，手指自然伸直",
                    "轻轻向下按动2-3次",
                    "动作柔和有力，表情诚恳"
                ],
                "magic_meaning": "🛡️ 释放保护护盾，寻求帮助和庇护",
                "sample_count": 0
            },
            "sick": {
                "chinese": "治疗术",
                "english": "Sick/Doctor",
                "sign_description": "右手食指中指按在左手腕，模拟把脉动作（'生病'手语）",
                "action_steps": [
                    "抬起左臂，手心朝上",
                    "右手食指中指并拢",
                    "轻轻按在左手腕脉搏处",
                    "保持2-3秒，模拟把脉"
                ],
                "magic_meaning": "💚 释放治疗魔法，治愈伤痛和疾病",
                "sample_count": 0
            },
            "danger": {
                "chinese": "闪电术",
                "english": "Danger",
                "sign_description": "双手食指交叉成X形在胸前摆动（'危险'手语）",
                "action_steps": [
                    "双手食指伸直，其他手指弯曲",
                    "食指交叉成X形",
                    "放在胸前位置",
                    "左右摆动2-3次，表情严肃"
                ],
                "magic_meaning": "⚡ 释放闪电攻击，警示危险和威胁",
                "sample_count": 0
            }
        }
        
        print("=" * 70)
        print("完整手语数据收集系统 v1.0")
        print("=" * 70)
    
    def extract_dual_hand_features(self, hand_landmarks_list, handedness):
        """提取双手特征"""
        left_hand_features = np.zeros(63)
        right_hand_features = np.zeros(63)
        
        if hand_landmarks_list:
            for i, hand_landmarks in enumerate(hand_landmarks_list):
                hand_type = 'Right'
                if handedness and i < len(handedness):
                    hand_type = handedness[i].classification[0].label
                
                features = []
                wrist = hand_landmarks.landmark[0]
                
                for lm in hand_landmarks.landmark:
                    features.extend([
                        lm.x - wrist.x,
                        lm.y - wrist.y,
                        lm.z - wrist.z
                    ])
                
                features_array = np.array(features)
                
                if hand_type == 'Left':
                    left_hand_features = features_array
                else:
                    right_hand_features = features_array
        
        combined_features = np.concatenate([left_hand_features, right_hand_features])
        return combined_features
    
    def draw_collection_interface(self, frame, sign_info, hands_detected, collecting):
        """
        绘制收集界面（支持中文）
        
        Args:
            frame: 视频帧
            sign_info: 手语信息
            hands_detected: 检测到的手数量
            collecting: 是否正在收集
            
        Returns:
            frame: 绘制后的帧
        """
        h, w = frame.shape[:2]
        
        # 1. 绘制标题和基本信息
        title = f"手语数据收集: {sign_info['chinese']} ({sign_info['english']})"
        frame = ChineseDisplaySupport.put_chinese_text(
            frame, title, (20, 30), font_size=24, color=(0, 255, 255)
        )
        
        # 2. 绘制魔法含义
        magic_text = f"魔法效果: {sign_info['magic_meaning']}"
        frame = ChineseDisplaySupport.put_chinese_text(
            frame, magic_text, (20, 70), font_size=18, color=(255, 200, 100)
        )
        
        # 3. 绘制动作描述
        desc_text = f"动作描述: {sign_info['sign_description']}"
        frame = ChineseDisplaySupport.put_chinese_text(
            frame, desc_text[:50], (20, 100), font_size=16, color=(200, 200, 255)
        )
        if len(desc_text) > 50:
            frame = ChineseDisplaySupport.put_chinese_text(
                frame, desc_text[50:], (20, 125), font_size=16, color=(200, 200, 255)
            )
        
        # 4. 绘制动作步骤指导
        y_offset = 160
        frame = ChineseDisplaySupport.put_chinese_text(
            frame, "动作步骤:", (20, y_offset), font_size=18, color=(100, 255, 100)
        )
        y_offset += 30
        
        for i, step in enumerate(sign_info['action_steps'], 1):
            step_text = f"{i}. {step}"
            frame = ChineseDisplaySupport.put_chinese_text(
                frame, step_text, (30, y_offset), font_size=14, color=(200, 255, 200)
            )
            y_offset += 25
        
        # 5. 绘制收集状态
        status_text = "状态: 正在收集中..." if collecting else "状态: 已暂停"
        status_color = (0, 255, 0) if collecting else (0, 0, 255)
        frame = ChineseDisplaySupport.put_chinese_text(
            frame, status_text, (20, y_offset + 10), font_size=20, color=status_color
        )
        
        # 6. 绘制样本计数
        count_text = f"样本进度: {self.counter}/{self.max_samples_per_gesture}"
        frame = ChineseDisplaySupport.put_chinese_text(
            frame, count_text, (20, y_offset + 40), font_size=18, color=(255, 255, 255)
        )
        
        # 7. 绘制进度条
        progress = self.counter / self.max_samples_per_gesture
        bar_width = 400
        bar_height = 20
        bar_x, bar_y = 20, y_offset + 70
        
        # 进度条背景
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        # 进度条前景
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + int(bar_width * progress), bar_y + bar_height), 
                     (0, 255, 0), -1)
        # 进度条边框
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 1)
        
        # 8. 绘制手部检测信息
        hands_status = f"手部检测: {hands_detected}只手"
        hands_color = (0, 255, 0) if hands_detected >= 2 else (0, 165, 255)
        frame = ChineseDisplaySupport.put_chinese_text(
            frame, hands_status, (20, bar_y + bar_height + 10), 
            font_size=16, color=hands_color
        )
        
        # 9. 绘制控制提示
        controls_y = h - 120
        controls = [
            "控制指令:",
            "S - 开始/暂停收集",
            "P - 暂停收集",
            "Q - 完成当前手语",
            "ESC - 退出系统"
        ]
        
        for i, control in enumerate(controls):
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            frame = ChineseDisplaySupport.put_chinese_text(
                frame, control, (20, controls_y + i * 25), 
                font_size=16, color=color
            )
        
        # 10. 绘制实时反馈
        if collecting and self.counter > 0:
            feedback_y = bar_y + bar_height + 40
            if hands_detected < 2:
                feedback = "⚠️ 请确保双手都在画面中"
                color = (0, 165, 255)
            elif progress < 0.3:
                feedback = "⏳ 请继续做出标准动作..."
                color = (255, 255, 0)
            elif progress < 0.7:
                feedback = "📊 数据收集中，保持动作..."
                color = (100, 255, 100)
            else:
                feedback = "✅ 数据收集良好，即将完成"
                color = (0, 255, 0)
            
            frame = ChineseDisplaySupport.put_chinese_text(
                frame, feedback, (20, feedback_y), font_size=16, color=color
            )
        
        return frame
    
    def collect_sign_gesture(self, sign_key):
        """
        收集特定手语的手势数据
        
        Args:
            sign_key: 手语键名
            
        Returns:
            success: 是否成功
        """
        if sign_key not in self.complete_sign_language:
            print(f"错误: 未知的手语 '{sign_key}'")
            return False
        
        sign_info = self.complete_sign_language[sign_key]
        self.current_label = sign_key
        self.collecting = False
        self.counter = 0
        
        print("\n" + "=" * 70)
        print(f"开始收集手语: {sign_info['chinese']}")
        print(f"英文: {sign_info['english']}")
        print(f"描述: {sign_info['sign_description']}")
        print(f"魔法效果: {sign_info['magic_meaning']}")
        print("=" * 70)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("摄像头已就绪，按 'S' 键开始收集数据...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
            
            # 水平翻转
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 手部检测
            results = self.hands.process(rgb_frame)
            
            # 检测到的手数量
            hands_detected = 0
            if results.multi_hand_landmarks:
                hands_detected = len(results.multi_hand_landmarks)
                
                # 绘制手部关键点
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            # 绘制中文界面
            frame = self.draw_collection_interface(frame, sign_info, hands_detected, self.collecting)
            
            # 数据收集逻辑
            if (self.collecting and results.multi_hand_landmarks and 
                self.counter < self.max_samples_per_gesture):
                
                # 控制采样频率（每5帧采样一次）
                if self.frame_count % 5 == 0:
                    if hands_detected >= 2:  # 确保检测到双手
                        features = self.extract_dual_hand_features(
                            results.multi_hand_landmarks,
                            results.multi_handedness if hasattr(results, 'multi_handedness') else []
                        )
                        
                        self.landmark_data.append(features)
                        self.labels.append(sign_key)
                        self.counter += 1
                        
                        # 在画面上显示收集成功提示
                        success_text = f"✓ 已收集 {self.counter} 个样本"
                        frame = ChineseDisplaySupport.put_chinese_text(
                            frame, success_text, (frame.shape[1] - 250, 50), 
                            font_size=18, color=(0, 255, 0)
                        )
                    else:
                        # 显示警告
                        warning_text = "⚠️ 需要检测到双手"
                        frame = ChineseDisplaySupport.put_chinese_text(
                            frame, warning_text, (frame.shape[1] - 250, 50), 
                            font_size=18, color=(0, 165, 255)
                        )
            
            self.frame_count += 1
            
            # 显示画面
            cv2.imshow("Sign Language Collection", frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                self.collecting = not self.collecting
                if self.collecting:
                    print("开始收集数据...")
                else:
                    print("暂停收集")
            
            elif key == ord('p') or key == ord('P'):
                self.collecting = False
                print("手动暂停")
            
            elif key == ord('q') or key == ord('Q'):
                print(f"完成 {sign_info['chinese']} 的数据收集")
                print(f"共收集 {self.counter} 个有效样本")
                break
            
            elif key == 27:  # ESC
                print("退出数据收集")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyWindow(f"手语数据收集 - {sign_info['chinese']}")
        
        # 更新样本计数
        sign_info['sample_count'] = self.counter
        
        return True
    
    def collect_all_signs_interactive(self):
        """
        交互式收集所有手语数据
        """
        print("\n完整手语数据收集系统")
        print("=" * 70)
        print("说明:")
        print("1. 本系统将帮助您收集4种真实手语数据")
        print("2. 每个手势需要收集约600个样本")
        print("3. 请按照标准手语动作进行录制")
        print("4. 建议在不同角度和光照下录制")
        print("=" * 70)
        
        print("\n需要收集的手语列表:")
        print("=" * 70)
        
        for key, info in self.complete_sign_language.items():
            print(f"\n【{info['chinese']}】 ({info['english']})")
            print(f"  描述: {info['sign_description']}")
            print(f"  魔法: {info['magic_meaning']}")
            print(f"  目标样本: {self.max_samples_per_gesture}")
        
        print("\n" + "=" * 70)
        
        # 收集每个手语
        collected_signs = []
        
        for sign_key in self.complete_sign_language.keys():
            sign_info = self.complete_sign_language[sign_key]
            
            print(f"\n准备收集: {sign_info['chinese']}")
            print(f"魔法效果: {sign_info['magic_meaning']}")
            
            # 询问是否收集
            response = input("开始收集这个手语？(y/n): ").strip().lower()
            if response != 'y':
                print(f"跳过 {sign_info['chinese']}")
                continue
            
            print(f"\n开始收集 {sign_info['chinese']}...")
            print("请参考以下动作步骤:")
            for step in sign_info['action_steps']:
                print(f"  {step}")
            
            input("\n按Enter键开始（确保摄像头准备就绪）...")
            
            success = self.collect_sign_gesture(sign_key)
            
            if not success:
                print("数据收集被中断")
                break
            
            collected_signs.append(sign_key)
            
            # 询问是否继续
            if sign_key != list(self.complete_sign_language.keys())[-1]:
                response = input(f"\n继续收集下一个手语？(y/n): ").strip().lower()
                if response != 'y':
                    break
        
        # 保存数据
        if len(self.landmark_data) > 0:
            self.save_complete_data(collected_signs)
        else:
            print("\n警告: 未收集到任何数据！")
    
    def save_complete_data(self, collected_signs):
        """
        保存完整的手语数据
        
        Args:
            collected_signs: 已收集的手语列表
        """
        if not collected_signs:
            print("错误: 没有数据可保存")
            return
        
        # 准备元数据
        metadata = {
            'collection_date': datetime.now().strftime("%Y-%m-%d"),
            'collection_time': datetime.now().strftime("%H:%M:%S"),
            'total_samples': len(self.landmark_data),
            'feature_dimension': len(self.landmark_data[0]) if self.landmark_data else 0,
            'sample_rate': '10 Hz',
            'hand_model': 'MediaPipe Hands v1.0',
            'signs_collected': collected_signs,
            'samples_per_sign': self.max_samples_per_gesture
        }
        
        # 准备保存的数据
        save_data = {
            'features': self.landmark_data,
            'labels': self.labels,
            'sign_language_info': self.complete_sign_language,
            'metadata': metadata
        }
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_sign_language_dataset_{timestamp}.pkl"
        save_path = os.path.join(self.data_dir, filename)
        
        # 保存数据
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print("\n" + "=" * 80)
        print("数据保存成功!")
        print("=" * 80)
        print(f"文件路径: {save_path}")
        print(f"总样本数: {len(self.landmark_data)}")
        print(f"特征维度: {metadata['feature_dimension']}")
        
        # 统计每个手语的样本数
        print("\n样本分布统计:")
        print("-" * 40)
        
        for sign_key in collected_signs:
            count = sum(1 for label in self.labels if label == sign_key)
            sign_name = self.complete_sign_language[sign_key]['chinese']
            percentage = (count / len(self.labels)) * 100
            print(f"{sign_name}: {count} 个样本 ({percentage:.1f}%)")
        
        print("\n下一步操作:")
        print("1. 运行训练脚本: python train_complete_model.py")
        print("2. 测试模型性能")
        print("3. 集成到主程序")
        print("=" * 80)
        
        return save_path

def main():
    """主函数"""
    print("真实手语数据收集系统")
    print("=" * 60)
    
    try:
        collector = CompleteSignLanguageCollector()
        collector.collect_all_signs_interactive()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n按Enter键退出程序...")

if __name__ == "__main__":
    main()