import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import random
import math
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import threading

# ======================================================
# 1. 增强字体处理函数（修复版）
# ======================================================
def get_safe_font(font_size):
    """获取安全的字体，避免方块显示"""
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
        "simhei.ttf",
        "arial.ttf"
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                
                return font
        except Exception as e:
            continue
    
    print("⚠️ 无法加载中文字体，使用默认字体")
    return ImageFont.load_default()

def put_cn_safe(img, text, pos, size=28, color=(255, 255, 255), background=None, 
                background_padding=(5, 5), max_width=None):
    """安全的文字绘制函数，支持自动换行和背景框（修复版）"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = get_safe_font(size)
        
        # 移除emoji
        text = text.replace("🎆", "").replace("🛡️", "").replace("⚡", "").replace("💊", "")
        text = text.replace("📖", "").replace("💡", "").replace("👐", "").replace("🔍", "")
        text = text.replace("✨", "").replace("🔒", "").replace("💀", "").replace("🎉", "")
        text = text.replace("🪄", "").replace("🎯", "").replace("🤖", "").replace("⏰", "")
        text = text.replace("⏹️", "").replace("🔋", "").replace("🔄", "").replace("👋", "")
        text = text.replace("💥", "").replace("💚", "").replace("📚", "").replace("⚠️", "")
        text = text.replace("✅", "").replace("❌", "").replace("📋", "").replace("📊", "")
        text = text.strip()
        
        # 如果指定最大宽度，处理自动换行
        if max_width:
            lines = []
            current_line = ""
            for char in text:
                test_line = current_line + char
                # 使用更稳定的方式计算文本宽度
                try:
                    # 尝试获取文本宽度
                    text_width = draw.textlength(test_line, font=font)
                    if text_width <= max_width:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = char
                except:
                    # 如果计算失败，简单按字符数分割
                    if len(test_line) * size / 2 <= max_width:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = char
            if current_line:
                lines.append(current_line)
        else:
            lines = [text]
        
        # 绘制每一行
        x, y = pos
        line_height = int(size * 1.2)
        
        for line in lines:
            # 计算文本大小（使用更稳定的方法）
            try:
                # 尝试获取边界框
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                # 如果失败，使用估计值
                text_width = len(line) * size // 2
                text_height = size
            
            # 绘制背景
            if background:
                # 如果background是3个值，添加透明度值
                if len(background) == 3:
                    bg_color = background + (255,)
                else:
                    bg_color = background
                
                # 创建背景矩形
                draw.rectangle(
                    [x - background_padding[0], y - background_padding[1],
                     x + text_width + background_padding[0], y + text_height + background_padding[1]],
                    fill=bg_color
                )
            
            # 绘制文本
            draw.text((x, y), line, font=font, fill=color, encoding="utf-8")
            
            y += line_height
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"⚠️ PIL绘制失败，使用OpenCV: {e}")
        # 如果PIL绘制失败，使用OpenCV
        for line in lines:
            cv2.putText(img, line, (int(pos[0]), int(pos[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, size/40, color, 2)
            pos = (pos[0], pos[1] + line_height)
        return img

# ======================================================
# 2. 增强音效系统
# ======================================================
try:
    import winsound
    
    class EnhancedSoundSystem:
        def __init__(self):
            self.enabled = True
            self.sounds = {
                "new_year": [(523, 100), (587, 100), (659, 100), (698, 100), (784, 100)],
                "shield": [(200, 200), (250, 150), (300, 100)],
                "lightning": [(800, 80), (600, 60), (400, 40)],
                "heal": [(392, 150), (440, 150), (493, 150)],
                "ui_start": (600, 100),
                "ui_end": (500, 80),
                "ui_success": (800, 150),
                "ui_fail": (300, 200),
                "attack": [(400, 100), (300, 80), (200, 60)],
                "damage": [(150, 200), (100, 150), (80, 100)],
                "heal_sound": [(261, 150), (329, 150), (392, 150)],
                "level_up": [(523, 200), (659, 150), (784, 100)],
                "enemy_defeated": [(150, 300), (100, 250), (80, 200)],
            }
        
        def play(self, sound_name, async_play=True):
            if not self.enabled or sound_name not in self.sounds:
                return
            
            sound_data = self.sounds[sound_name]
            
            def play_thread():
                try:
                    if isinstance(sound_data[0], tuple):
                        for freq, duration in sound_data:
                            winsound.Beep(freq, duration)
                            time.sleep(0.03)
                    else:
                        freq, duration = sound_data
                        winsound.Beep(freq, duration)
                except:
                    pass
            
            if async_play:
                thread = threading.Thread(target=play_thread)
                thread.daemon = True
                thread.start()
            else:
                play_thread()
    
    sound = EnhancedSoundSystem()
    
except ImportError:
    class EnhancedSoundSystem:
        def play(self, *args, **kwargs): pass
    sound = EnhancedSoundSystem()

# ======================================================
# 3. 增强魔法系统配置（移除emoji）
# ======================================================
SPELL_CONFIG = {
    "new_year": {
        "name": "庆祝咒",
        "color": (255, 200, 50),
        "mana_cost": 15,
        "gesture_label": "celebrate",
        "damage": 20,
        "healing": 0,
        "cooldown": 2.0,
        "description": "释放烟花庆祝，对敌人造成AOE伤害",
        "effect_type": "explosion"
    },
    "shield": {
        "name": "护盾术", 
        "color": (100, 200, 255),
        "mana_cost": 20,
        "gesture_label": "help",
        "damage": 15,
        "healing": 0,
        "cooldown": 3.0,
        "description": "召唤护盾保护自己，同时反弹伤害",
        "effect_type": "shield"
    },
    "lightning": {
        "name": "闪电术",
        "color": (255, 255, 100),
        "mana_cost": 25,
        "gesture_label": "danger",
        "damage": 30,
        "healing": 0,
        "cooldown": 4.0,
        "description": "召唤闪电打击敌人，造成高额伤害",
        "effect_type": "lightning"
    },
    "heal": {
        "name": "治疗术",
        "color": (100, 255, 150),
        "mana_cost": 30,
        "gesture_label": "sick",
        "damage": 0,
        "healing": 40,
        "cooldown": 5.0,
        "description": "释放治愈能量，恢复生命值",
        "effect_type": "heal"
    }
}

# ======================================================
# 4. 敌人系统（美观魔法角色版，移除emoji）
# ======================================================
class Enemy:
    def __init__(self, name="暗影巫师", max_health=200):
        self.name = name
        self.health = max_health
        self.max_health = max_health
        self.position = (0.7, 0.35)  # 调整位置到屏幕右侧偏中
        self.last_attacked = 0
        self.hurt_effect = 0
        self.heal_effect = 0
        self.status = "alive"
        self.attack_cooldown = 0
        self.last_attack_time = 0
        
    def take_damage(self, damage):
        if self.health > 0 and self.status != "defeated":
            self.health = max(0, self.health - damage)
            self.hurt_effect = 1.0
            sound.play("damage")
            print(f"敌人受到 {damage} 点伤害，剩余生命: {self.health}")
            
            if self.health <= 0:
                self.status = "defeated"
                sound.play("enemy_defeated")
                print(f"敌人 {self.name} 已被击败！")
            return damage
        return 0
    
    def heal(self, amount):
        if self.status != "defeated":
            old_health = self.health
            self.health = min(self.max_health, self.health + amount)
            healed = self.health - old_health
            if healed > 0:
                self.heal_effect = 1.0
                sound.play("heal_sound")
                print(f"敌人恢复 {healed} 点生命，当前生命: {self.health}")
            return healed
        return 0
    
    def update(self, delta_time):
        if self.hurt_effect > 0:
            self.hurt_effect -= delta_time * 3
        if self.heal_effect > 0:
            self.heal_effect -= delta_time * 2
        
        # 更新攻击冷却
        if self.attack_cooldown > 0:
            self.attack_cooldown -= delta_time
        
        # 确保生命值不为负
        self.health = max(0, min(self.max_health, self.health))
        
        # 更新状态
        if self.health <= 0:
            self.status = "defeated"
    
    def draw(self, frame, frame_width, frame_height):
        # 计算敌人在屏幕上的绝对位置
        x = int(frame_width * self.position[0])
        y = int(frame_height * self.position[1])
        
        # 确保敌人不会绘制在UI面板上
        panel_width = 300
        min_x = 100
        max_x = frame_width - panel_width - 150
        x = max(min_x, min(max_x, x))
        
        # 确保y坐标在合理范围内
        y = max(150, min(frame_height - 200, y))
        
        # 绘制敌人背景光环
        if self.status != "defeated":
            # 动态光环效果
            halo_size = 80 + int(10 * math.sin(time.time() * 3))
            if self.hurt_effect > 0:
                halo_color = (255, 100, 100, 100)  # 受伤时红色光环
            elif self.heal_effect > 0:
                halo_color = (100, 255, 100, 100)  # 治疗时绿色光环
            else:
                halo_color = (150, 100, 255, 100)  # 正常紫色光环
            
            # 绘制多层光环
            for i in range(3, 0, -1):
                halo_radius = halo_size + i * 10
                alpha = 0.3 / i
                halo_overlay = frame.copy()
                cv2.circle(halo_overlay, (x, y), halo_radius, 
                          halo_color[:3], -1)
                cv2.addWeighted(halo_overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 绘制魔法师角色
        # 1. 身体（长袍）
        robe_height = 100
        robe_width = 60
        robe_top = y - robe_height // 2
        robe_color = (120, 60, 180) if self.status != "defeated" else (80, 80, 80)
        
        # 长袍主体
        pts = np.array([
            [x - robe_width//2, robe_top],
            [x + robe_width//2, robe_top],
            [x + robe_width//3, robe_top + robe_height],
            [x - robe_width//3, robe_top + robe_height]
        ], np.int32)
        cv2.fillPoly(frame, [pts], robe_color)
        
        # 长袍装饰边
        if self.status != "defeated":
            cv2.polylines(frame, [pts], True, (200, 150, 255), 2)
        
        # 2. 头部
        head_radius = 25
        head_color = (240, 220, 180) if self.status != "defeated" else (180, 180, 180)
        cv2.circle(frame, (x, robe_top - 5), head_radius, head_color, -1)
        cv2.circle(frame, (x, robe_top - 5), head_radius, (50, 30, 80), 2)
        
        # 3. 面部特征
        if self.status == "defeated":
            # 被击败时打叉的眼睛
            eye_y = robe_top - 10
            cv2.line(frame, (x - 10, eye_y - 5), (x - 5, eye_y), (255, 50, 50), 2)
            cv2.line(frame, (x - 5, eye_y - 5), (x - 10, eye_y), (255, 50, 50), 2)
            cv2.line(frame, (x + 10, eye_y - 5), (x + 5, eye_y), (255, 50, 50), 2)
            cv2.line(frame, (x + 5, eye_y - 5), (x + 10, eye_y), (255, 50, 50), 2)
            
            # "X"形嘴巴
            mouth_y = robe_top + 5
            cv2.line(frame, (x - 8, mouth_y - 3), (x + 8, mouth_y + 3), (255, 50, 50), 2)
            cv2.line(frame, (x + 8, mouth_y - 3), (x - 8, mouth_y + 3), (255, 50, 50), 2)
        else:
            # 正常眼睛
            eye_y = robe_top - 10
            eye_color = (255, 255, 255)
            pupil_color = (60, 30, 100)
            
            # 左眼
            cv2.circle(frame, (x - 8, eye_y), 6, eye_color, -1)
            cv2.circle(frame, (x - 8, eye_y), 6, (50, 30, 80), 1)
            cv2.circle(frame, (x - 8, eye_y), 3, pupil_color, -1)
            
            # 右眼
            cv2.circle(frame, (x + 8, eye_y), 6, eye_color, -1)
            cv2.circle(frame, (x + 8, eye_y), 6, (50, 30, 80), 1)
            cv2.circle(frame, (x + 8, eye_y), 3, pupil_color, -1)
            
            # 嘴巴（根据生命值变化）
            mouth_y = robe_top + 5
            if self.health < 50:
                # 痛苦表情
                cv2.ellipse(frame, (x, mouth_y), (10, 6), 0, 0, 180, (50, 30, 80), 2)
            else:
                # 邪恶笑容
                cv2.ellipse(frame, (x, mouth_y), (10, 4), 0, 0, 180, (50, 30, 80), 2)
        
        # 4. 魔法帽
        if self.status != "defeated":
            hat_height = 40
            hat_width = 50
            hat_top = robe_top - head_radius - 15
            
            # 帽檐
            cv2.ellipse(frame, (x, hat_top + 10), (hat_width//2, 8), 0, 0, 360, (80, 40, 120), -1)
            
            # 帽顶
            hat_pts = np.array([
                [x - hat_width//3, hat_top + 10],
                [x + hat_width//3, hat_top + 10],
                [x, hat_top - hat_height]
            ], np.int32)
            cv2.fillPoly(frame, [hat_pts], (100, 50, 150))
            cv2.polylines(frame, [hat_pts], True, (150, 100, 200), 2)
            
            # 帽顶星星
            star_size = 8
            star_color = (255, 255, 100)
            for angle in range(0, 360, 72):
                rad = math.radians(angle)
                star_x = int(x + star_size * 0.5 * math.cos(rad))
                star_y = int(hat_top - hat_height + 5 + star_size * 0.5 * math.sin(rad))
                cv2.circle(frame, (star_x, star_y), 2, star_color, -1)
        
        # 5. 魔法杖
        if self.status != "defeated":
            staff_length = 80
            staff_end_x = x + robe_width//2 + 20
            staff_end_y = y
            
            # 杖身
            cv2.line(frame, (x + 15, robe_top + 30), 
                    (staff_end_x, staff_end_y), 
                    (150, 120, 80), 4)
            
            # 杖头（水晶球）
            cv2.circle(frame, (staff_end_x, staff_end_y), 12, (200, 220, 255), -1)
            cv2.circle(frame, (staff_end_x, staff_end_y), 12, (100, 150, 255), 2)
            
            # 水晶球内部光点
            for i in range(3):
                light_x = staff_end_x + random.randint(-5, 5)
                light_y = staff_end_y + random.randint(-5, 5)
                cv2.circle(frame, (light_x, light_y), 2, (255, 255, 255), -1)
        
        # 生命条（美观设计）
        bar_width = 140
        bar_height = 16
        bar_x = x - bar_width // 2
        bar_y = robe_top + robe_height + 30
        
        # 生命条背景（带装饰）
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (40, 30, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (100, 80, 120), 2)
        
        # 生命条装饰端点
        cv2.circle(frame, (bar_x, bar_y + bar_height//2), bar_height//2, (100, 80, 120), -1)
        cv2.circle(frame, (bar_x + bar_width, bar_y + bar_height//2), bar_height//2, (100, 80, 120), -1)
        
        # 生命值填充（渐变效果）
        health_percent = self.health / self.max_health
        fill_width = int(bar_width * health_percent)
        
        if health_percent > 0.7:
            # 绿色到青色渐变
            for i in range(fill_width):
                color_ratio = i / fill_width
                r = int(100 - 100 * color_ratio)
                g = int(200 + 55 * color_ratio)
                b = int(150 - 50 * color_ratio)
                cv2.line(frame, (bar_x + i, bar_y + 2), 
                        (bar_x + i, bar_y + bar_height - 2), 
                        (b, g, r), 1)
        elif health_percent > 0.3:
            # 橙色渐变
            for i in range(fill_width):
                color_ratio = i / fill_width
                r = int(255 - 55 * color_ratio)
                g = int(165 + 40 * color_ratio)
                b = int(0 + 100 * color_ratio)
                cv2.line(frame, (bar_x + i, bar_y + 2), 
                        (bar_x + i, bar_y + bar_height - 2), 
                        (b, g, r), 1)
        else:
            # 红色渐变
            for i in range(fill_width):
                color_ratio = i / fill_width
                r = int(255 - 100 * color_ratio)
                g = int(50 + 30 * color_ratio)
                b = int(50 + 50 * color_ratio)
                cv2.line(frame, (bar_x + i, bar_y + 2), 
                        (bar_x + i, bar_y + bar_height - 2), 
                        (b, g, r), 1)
        
        # 生命条边框高光
        cv2.rectangle(frame, (bar_x + 1, bar_y + 1), 
                     (bar_x + bar_width - 1, bar_y + bar_height - 1), 
                     (180, 160, 200), 1)
        
        # 敌人名字
        name_y = bar_y - 25
        name_color = (255, 220, 180) if self.status != "defeated" else (180, 180, 180)
        frame = put_cn_safe(frame, self.name, 
                           (x - len(self.name) * 9, name_y), 
                           20, name_color)
        
        # 生命值文本
        health_text = f"{int(self.health)}/{self.max_health}"
        health_text_color = (255, 255, 255) if self.status != "defeated" else (150, 150, 150)
        
        frame = put_cn_safe(frame, health_text, 
                           (x - len(health_text) * 5, bar_y + bar_height + 20), 
                           16, health_text_color)
        
        # 状态指示器
        if self.status == "defeated":
            status_text = "已击败"
            status_color = (255, 100, 100)
            status_y = bar_y + bar_height + 45
            
            frame = put_cn_safe(frame, status_text, 
                               (x - 20, status_y), 
                               18, status_color)
        
        # 显示伤害/治疗特效
        elif self.hurt_effect > 0.7:
            hurt_text = f"-{int(self.hurt_effect * 15)}"
            hurt_y = robe_top - 40
            cv2.putText(frame, hurt_text, (x - 20, hurt_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2)
        elif self.heal_effect > 0.7:
            heal_text = f"+{int(self.heal_effect * 20)}"
            heal_y = robe_top - 40
            cv2.putText(frame, heal_text, (x - 20, heal_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        
        return frame

# ======================================================
# 5. 增强手势识别类
# ======================================================
class EnhancedGestureRecognizer:
    def __init__(self, model_path="trained_models\\latest_sign_language_model.joblib"):
        print("正在加载增强手势识别模型...")
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.scaler = model_data['scaler']
            self.sign_info = model_data.get('sign_info', {})
            
            print("手势识别模型加载成功")
            
            print("模型支持的标签:")
            for label in self.label_encoder.classes_:
                sign_name = self.sign_info.get(label, {}).get('chinese', label)
                print(f"  - {label} -> {sign_name}")
            
            print(f"模型类型: {type(self.model).__name__}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def extract_features(self, hand_landmarks_list, handedness):
        """提取特征（与训练代码完全一致）"""
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
    
    def recognize(self, hand_landmarks_list, handedness=None, debug=False):
        """识别手势"""
        if self.model is None:
            return None, 0.0, []
        
        if not hand_landmarks_list or len(hand_landmarks_list) < 2:
            return None, 0.0, []
        
        try:
            features = self.extract_features(hand_landmarks_list, handedness)
            
            if len(features) != 126:
                return None, 0.0, []
            
            features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            pred_proba = self.model.predict_proba(features_scaled)[0]
            pred_idx = np.argmax(pred_proba)
            confidence = pred_proba[pred_idx]
            
            # 获取所有预测概率
            all_probs = []
            for i, (cls, prob) in enumerate(zip(self.label_encoder.classes_, pred_proba)):
                all_probs.append((cls, prob))
            
            # 按概率排序
            all_probs.sort(key=lambda x: x[1], reverse=True)
            
            if debug:
                print("\n" + "=" * 60)
                print("预测概率分布 (降序):")
                for cls, prob in all_probs:
                    sign_name = self.sign_info.get(cls, {}).get('chinese', cls)
                    print(f"  {cls} ({sign_name}): {prob:.4f}")
                print(f"\n最高置信度: {confidence:.4f}")
                print("=" * 60)
            
            if confidence >= 0.4:
                pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
                return pred_label, confidence, all_probs
            
            return None, confidence, all_probs
            
        except Exception as e:
            if debug:
                print(f"识别错误: {e}")
            return None, 0.0, []

# ======================================================
# 6. 增强魔法系统（移除emoji）
# ======================================================
class EnhancedMagicSystem:
    def __init__(self, enemy):
        self.mana = 100.0
        self.max_mana = 100.0
        self.mana_regen = 0.8
        
        # 修复：所有法术默认都学会
        self.spells_learned = {
            "new_year": True,
            "shield": True,
            "lightning": True,
            "heal": True
        }
        
        self.combo = 0
        self.last_cast_time = 0
        self.cast_history = []
        self.enemy = enemy
        self.player_health = 100
        self.max_player_health = 100
        
        # 标签映射
        self.label_to_spell = {
            "celebrate": "new_year",
            "help": "shield",
            "sick": "heal",
            "danger": "lightning"
        }
        
        print("初始化增强魔法系统...")
        print("手势到法术的映射关系:")
        for label, spell in self.label_to_spell.items():
            spell_name = SPELL_CONFIG.get(spell, {}).get('name', spell)
            learned = self.spells_learned.get(spell, False)
            status = "已学会" if learned else "未学会"
            print(f"  {label} -> {spell_name} ({status})")
    
    def update(self, delta_time):
        self.mana = min(self.max_mana, self.mana + self.mana_regen * delta_time)
        
        current_time = time.time()
        if current_time - self.last_cast_time > 5.0:
            self.combo = max(0, self.combo - 1)
    
    def can_cast(self, spell_type):
        spell_info = SPELL_CONFIG.get(spell_type, {})
        if not self.spells_learned.get(spell_type, False):
            return False, f"尚未学会 {spell_info.get('name', '此法术')}"
        
        mana_cost = spell_info.get("mana_cost", 20)
        if self.mana < mana_cost:
            return False, "魔力不足"
        
        current_time = time.time()
        last_cast_time = 0
        for cast in reversed(self.cast_history):
            if cast["spell"] == spell_type:
                last_cast_time = cast["time"]
                break
        
        cooldown = spell_info.get("cooldown", 2.0)
        if current_time - last_cast_time < cooldown:
            remaining = cooldown - (current_time - last_cast_time)
            return False, f"冷却中 ({remaining:.1f}秒)"
        
        return True, ""
    
    def cast_spell(self, spell_type):
        success, message = self.can_cast(spell_type)
        if not success:
            return False, message, 0, 0
        
        spell_info = SPELL_CONFIG.get(spell_type, {})
        mana_cost = spell_info.get("mana_cost", 20)
        self.mana -= mana_cost
        
        current_time = time.time()
        if current_time - self.last_cast_time < 3.0:
            self.combo += 1
        else:
            self.combo = 1
        
        self.last_cast_time = current_time
        
        # 记录施法历史
        self.cast_history.append({
            "spell": spell_type,
            "time": current_time,
            "damage": spell_info.get("damage", 0),
            "healing": spell_info.get("healing", 0)
        })
        
        # 保留最近10次施法记录
        if len(self.cast_history) > 10:
            self.cast_history.pop(0)
        
        # 自动学习新法术（如果未学会）
        if not self.spells_learned.get(spell_type, False):
            self.spells_learned[spell_type] = True
            sound.play("level_up")
            print(f"学会了新法术: {spell_info.get('name', '法术')}")
        
        # 处理伤害和治疗
        damage_dealt = 0
        healing_done = 0
        
        if spell_type == "heal":
            # 治疗术治疗敌人
            healing_done = self.enemy.heal(spell_info.get("healing", 0))
            sound.play("heal_sound")
        else:
            # 其他法术造成伤害
            damage_dealt = self.enemy.take_damage(spell_info.get("damage", 0))
            sound.play("attack")
        
        return True, f"{spell_info.get('name', '法术')} 施放成功！", damage_dealt, healing_done

# ======================================================
# 7. 增强魔法特效系统（移除emoji）
# ======================================================
class EnhancedSpellEffect:
    def __init__(self, spell_type, center, confidence, frame_width, frame_height):
        self.spell_type = spell_type
        self.center = center
        self.confidence = confidence
        self.start_time = time.time()
        self.particles = []
        self.effect_frames = []
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        config = SPELL_CONFIG.get(self.spell_type, SPELL_CONFIG["new_year"])
        color = config["color"]
        
        # 根据法术类型创建不同特效
        if self.spell_type == "new_year":
            self.create_firework_effect(color)
        elif self.spell_type == "shield":
            self.create_shield_effect(color)
        elif self.spell_type == "lightning":
            self.create_lightning_effect(color)
        elif self.spell_type == "heal":
            self.create_heal_effect(color)
    
    def create_firework_effect(self, color):
        """烟花特效"""
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            radius = random.uniform(3, 8)
            
            self.particles.append({
                "x": self.center[0],
                "y": self.center[1],
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "size": radius,
                "color": color,
                "life": random.uniform(1.0, 2.0),
                "born": time.time(),
                "trail": []
            })
    
    def create_shield_effect(self, color):
        """护盾特效"""
        for i in range(36):
            angle = (i * 10) * math.pi / 180
            radius = 80 + random.uniform(-10, 10)
            
            self.particles.append({
                "x": self.center[0] + math.cos(angle) * radius,
                "y": self.center[1] + math.sin(angle) * radius,
                "size": random.uniform(4, 8),
                "color": color,
                "life": random.uniform(1.5, 2.5),
                "born": time.time(),
                "angle": angle,
                "radius": radius,
                "rotation": random.uniform(0.05, 0.15)
            })
    
    def create_lightning_effect(self, color):
        """闪电特效"""
        # 创建闪电主干
        segments = 20
        start_x = self.center[0] - 100
        end_x = self.center[0] + 100
        start_y = self.center[1] - 150
        end_y = self.center[1] + 50
        
        for i in range(segments):
            t = i / segments
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t
            
            # 添加随机抖动
            x += random.uniform(-15, 15)
            y += random.uniform(-10, 10)
            
            self.particles.append({
                "x": x,
                "y": y,
                "size": random.uniform(8, 15),
                "color": color,
                "life": random.uniform(0.3, 0.7),
                "born": time.time(),
                "segment": i
            })
        
        # 添加闪电分支
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            length = random.uniform(30, 80)
            
            self.particles.append({
                "x": self.center[0] + math.cos(angle) * length,
                "y": self.center[1] + math.sin(angle) * length,
                "size": random.uniform(3, 6),
                "color": (255, 255, 200),
                "life": random.uniform(0.2, 0.4),
                "born": time.time()
            })
    
    def create_heal_effect(self, color):
        """治疗特效"""
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(20, 60)
            speed = random.uniform(0.5, 1.5)
            
            self.particles.append({
                "x": self.center[0],
                "y": self.center[1],
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "size": random.uniform(6, 12),
                "color": color,
                "life": random.uniform(1.5, 2.5),
                "born": time.time(),
                "radius": radius,
                "upward": random.uniform(0.5, 1.5)
            })
    
    def update(self, frame):
        elapsed = time.time() - self.start_time
        current_time = time.time()
        
        new_particles = []
        config = SPELL_CONFIG.get(self.spell_type, {})
        
        for p in self.particles:
            life_left = 1 - (current_time - p["born"]) / p["life"]
            
            if life_left > 0:
                # 更新位置
                if "vx" in p and "vy" in p:
                    p["x"] += p["vx"]
                    p["y"] += p["vy"]
                    
                    if self.spell_type == "heal" and "upward" in p:
                        p["vy"] -= p["upward"] * 0.1
                
                # 特殊效果更新
                if self.spell_type == "shield" and "angle" in p:
                    p["angle"] += p.get("rotation", 0.1)
                    p["x"] = self.center[0] + math.cos(p["angle"]) * p["radius"]
                    p["y"] = self.center[1] + math.sin(p["angle"]) * p["radius"]
                
                # 绘制粒子
                alpha = life_left
                if self.spell_type == "lightning":
                    alpha = min(1.0, life_left * 2)  # 闪电更亮
                
                color = (
                    int(p["color"][0] * alpha),
                    int(p["color"][1] * alpha),
                    int(p["color"][2] * alpha)
                )
                
                # 绘制粒子
                if self.spell_type == "lightning":
                    # 闪电连接线
                    if "segment" in p and p["segment"] > 0:
                        prev_p = self.particles[p["segment"] - 1]
                        if (current_time - prev_p["born"]) / prev_p["life"] < 1:
                            cv2.line(frame, 
                                    (int(p["x"]), int(p["y"])),
                                    (int(prev_p["x"]), int(prev_p["y"])),
                                    color, 3)
                    
                    cv2.circle(frame, (int(p["x"]), int(p["y"])),
                              int(p["size"]), color, -1)
                else:
                    cv2.circle(frame, (int(p["x"]), int(p["y"])),
                              int(p["size"]), color, -1)
                
                # 添加辉光效果
                if random.random() < 0.3:
                    glow_size = int(p["size"] * 1.5)
                    glow_color = tuple(min(255, c + 50) for c in color)
                    cv2.circle(frame, (int(p["x"]), int(p["y"])),
                              glow_size, glow_color, 1)
                
                new_particles.append(p)
        
        self.particles = new_particles
        
        # 显示法术名称和效果
        if elapsed < 2.0:
            spell_name = config.get("name", "法术")
            description = config.get("description", "")
            
            # 法术名称
            name_size = 32
            name_x = self.center[0] - 80
            name_y = self.center[1] - 100
            
            frame = put_cn_safe(frame, spell_name,
                               (name_x, name_y),
                               name_size, config.get("color", (255,255,255)),
                               background=(20, 20, 20),
                               background_padding=(10, 5))
            
            # 法术描述
            if elapsed > 0.5 and elapsed < 1.5:
                desc_y = name_y + 50
                frame = put_cn_safe(frame, description,
                                   (name_x, desc_y),
                                   18, (200, 220, 255),
                                   background=(30, 30, 40),
                                   background_padding=(10, 5),
                                   max_width=200)
        
        return elapsed < 3.0

# ======================================================
# 8. 增强UI绘制函数（优化版，移除小框和emoji）
# ======================================================
def draw_enhanced_magic_ui(frame, magic_system, frame_width, frame_height, 
                          gesture_label=None, confidence=0, recording=False,
                          recording_frames=None, current_phase="idle"):
    """绘制增强魔法系统UI（优化版）"""
    h, w = frame_height, frame_width
    
    # ===== 左侧法术书面板 =====
    panel_width = 280  # 保持与原始代码一致
    panel_x = 10
    panel_y_top = 80
    panel_y_bottom = h - 10
    
    # 面板背景（不透明，深色）
    cv2.rectangle(frame, (panel_x, panel_y_top), (panel_x + panel_width, panel_y_bottom),
                 (30, 25, 40), -1)
    
    # 面板边框
    cv2.rectangle(frame, (panel_x, panel_y_top), (panel_x + panel_width, panel_y_bottom),
                 (100, 80, 120), 2)
    
    # 面板标题
    title_x = panel_x + 20
    title_y = panel_y_top + 30
    frame = put_cn_safe(frame, "魔法法典", (title_x, title_y), 
                       24, (255, 240, 200))
    
    # 标题下划线
    cv2.line(frame, (title_x, title_y + 25), (title_x + 100, title_y + 25),
            (180, 160, 220), 1)
    
    # 玩家生命条
    player_health_y = title_y + 40
    player_health_width = panel_width - 40
    health_percent = magic_system.player_health / magic_system.max_player_health
    
    # 生命条背景
    cv2.rectangle(frame, (panel_x + 20, player_health_y), 
                 (panel_x + 20 + player_health_width, player_health_y + 20),
                 (40, 25, 30), -1)
    cv2.rectangle(frame, (panel_x + 20, player_health_y), 
                 (panel_x + 20 + player_health_width, player_health_y + 20),
                 (100, 70, 80), 1)
    
    # 生命值填充
    fill_width = int(player_health_width * health_percent)
    health_color = (0, 255, 0) if health_percent > 0.6 else (
        (255, 165, 0) if health_percent > 0.3 else (255, 0, 0))
    
    cv2.rectangle(frame, (panel_x + 20, player_health_y), 
                 (panel_x + 20 + fill_width, player_health_y + 20),
                 health_color, -1)
    
    # 生命值文本
    health_text = f"生命值: {int(magic_system.player_health)}/{int(magic_system.max_player_health)}"
    frame = put_cn_safe(frame, health_text, (panel_x + 25, player_health_y + 15), 
                       18, (255, 255, 255))
    
    # 魔力条
    mana_y = player_health_y + 35
    mana_percent = magic_system.mana / magic_system.max_mana
    
    # 魔力条背景
    cv2.rectangle(frame, (panel_x + 20, mana_y), 
                 (panel_x + 20 + player_health_width, mana_y + 20),
                 (25, 25, 40), -1)
    cv2.rectangle(frame, (panel_x + 20, mana_y), 
                 (panel_x + 20 + player_health_width, mana_y + 20),
                 (80, 80, 120), 1)
    
    # 魔力值填充
    fill_width = int(player_health_width * mana_percent)
    if mana_percent > 0.5:
        bar_color = (100, 200, 255)
    elif mana_percent > 0.2:
        bar_color = (255, 200, 100)
    else:
        bar_color = (255, 100, 100)
    
    cv2.rectangle(frame, (panel_x + 20, mana_y), 
                 (panel_x + 20 + fill_width, mana_y + 20),
                 bar_color, -1)
    
    # 魔力值文本
    mana_text = f"魔力值: {int(magic_system.mana)}/{int(magic_system.max_mana)}"
    frame = put_cn_safe(frame, mana_text, (panel_x + 25, mana_y + 15), 
                       18, (255, 255, 255))
    
    # 连击显示
    if magic_system.combo > 0:
        combo_x = panel_x + panel_width // 2 - 30
        combo_y = mana_y + 40
        combo_color = (255, 200 + min(55, magic_system.combo * 5), 
                      100 + min(155, magic_system.combo * 10))
        
        combo_text = f"连击 x{magic_system.combo}"
        frame = put_cn_safe(frame, combo_text, (combo_x, combo_y), 
                           22 + min(8, magic_system.combo), combo_color)
    
    # 法术列表标题
    spells_y = mana_y + 80
    frame = put_cn_safe(frame, "已学法术:", (panel_x + 20, spells_y), 
                       22, (220, 220, 255))
    
    spells_y += 35
    spells_displayed = 0
    
    for spell_id, learned in magic_system.spells_learned.items():
        if spells_displayed >= 4:
            break
            
        spell_info = SPELL_CONFIG.get(spell_id, {})
        spell_name = spell_info.get("name", spell_id)
        spell_color = spell_info.get("color", (200, 200, 200))
        
        if learned:
            spell_text = f"  {spell_name}"
            text_color = spell_color
        else:
            spell_text = "  ???? (未学会)"
            text_color = (150, 150, 150)
        
        # 检查冷却
        current_time = time.time()
        last_cast_time = 0
        for cast in reversed(magic_system.cast_history):
            if cast["spell"] == spell_id:
                last_cast_time = cast["time"]
                break
        
        cooldown = spell_info.get("cooldown", 2.0)
        if last_cast_time > 0 and (current_time - last_cast_time) < cooldown:
            remaining = cooldown - (current_time - last_cast_time)
            cooldown_text = f" ({remaining:.1f}s)"
            spell_text += cooldown_text
            text_color = tuple(c // 2 for c in text_color)
        
        frame = put_cn_safe(frame, spell_text, (panel_x + 25, spells_y), 
                           20, text_color)
        spells_y += 30
        spells_displayed += 1
    
    # ===== 右侧信息面板 =====
    info_panel_width = 320  # 保持与原始代码一致
    info_panel_x = w - info_panel_width - 10
    
    # 操作说明面板背景
    cv2.rectangle(frame, (info_panel_x, 80), (w - 10, 285),
                 (40, 30, 50), -1)
    cv2.rectangle(frame, (info_panel_x, 80), (w - 10, 285),
                 (150, 120, 200), 2)
    
    # 面板标题
    frame = put_cn_safe(frame, "操作指南", (info_panel_x + 25, 100), 
                       24, (255, 240, 200))
    
   
    
    instructions = [
        "S: 开始录制手势",
        "E: 结束录制/攻击",
        "M: 恢复魔力",
        "R: 重置连击",
        "D: 调试模式",
        "ESC: 退出系统"
    ]
    
    inst_y = 135
    for inst in instructions:
        frame = put_cn_safe(frame, inst, (info_panel_x + 35, inst_y), 
                          18, (220, 230, 255))
        inst_y += 25
    
    # 手势识别面板
    gesture_panel_y = 285
    gesture_panel_height = 300
    
    # 面板背景
    cv2.rectangle(frame, (info_panel_x, gesture_panel_y), 
                 (w - 10, gesture_panel_y + gesture_panel_height),
                 (45, 35, 55), -1)
    cv2.rectangle(frame, (info_panel_x, gesture_panel_y), 
                 (w - 10, gesture_panel_y + gesture_panel_height),
                 (180, 140, 220), 2)
    
    # 面板标题
    frame = put_cn_safe(frame, "手语动作", (info_panel_x + 25, gesture_panel_y + 25), 
                       22, (255, 240, 200))
    
    # 手语动作列表（简洁显示，无小框）
    gestures_info = [
        ("庆祝咒 (celebrate)", "左手横放，右手竖起大拇指滑动"),
        ("护盾术 (help)", "双手斜伸掌心向外"),
        ("闪电术 (danger)", "双手食指交叉成X形"),
        ("治疗术 (sick)", "两指按手腕把脉")
    ]
    
    gest_y = gesture_panel_y + 60
    for gesture_name, gesture_desc in gestures_info:
        # 手势名称
        frame = put_cn_safe(frame, gesture_name, 
                           (info_panel_x + 25, gest_y), 
                           20, (255, 220, 180))
        
        # 手势描述
        frame = put_cn_safe(frame, gesture_desc, 
                           (info_panel_x + 30, gest_y + 30), 
                           18, (200, 220, 255),
                           max_width=280)
        
        gest_y += 60
    
    # 实时检测显示
    if gesture_label and confidence > 0.3:
        detect_panel_y = gesture_panel_y + gesture_panel_height + 15
        detect_panel_height = 100
        
        # 检测面板背景
        cv2.rectangle(frame, (info_panel_x, detect_panel_y), 
                     (w - 10, detect_panel_y + detect_panel_height),
                     (50, 40, 60), -1)
        cv2.rectangle(frame, (info_panel_x, detect_panel_y), 
                     (w - 10, detect_panel_y + detect_panel_height),
                     (200, 180, 240), 2)
        
        # 映射到法术
        spell_type = magic_system.label_to_spell.get(gesture_label)
        if spell_type:
            spell_name = SPELL_CONFIG.get(spell_type, {}).get("name", "未知")
            detect_text = f"检测到: {spell_name}"
        else:
            detect_text = f"检测到: {gesture_label}"
        
        # 绘制检测结果
        text_x = info_panel_x + 25
        text_y = detect_panel_y + 25
        
        frame = put_cn_safe(frame, detect_text, 
                           (text_x, text_y), 
                           20, (255, 200, 100))
        
        conf_text = f"置信度: {confidence:.3f}"
        frame = put_cn_safe(frame, conf_text, 
                           (text_x, text_y + 30), 
                           18, (200, 255, 200))
    
    # ===== 底部录制进度条 =====
    if recording and recording_frames is not None:
        progress_y = h - 100
        progress_width = 400
        progress_x = w // 2 - progress_width // 2
        
        # 进度条背景
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_width, progress_y + 30),
                     (60, 60, 80), -1)
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_width, progress_y + 30),
                     (120, 140, 180), 2)
        
        # 动态进度条效果
        progress = min(1.0, len(recording_frames) / 40.0)
        filled_width = int(progress_width * progress)
        
        # 渐变色进度条
        for i in range(filled_width):
            color_ratio = i / progress_width
            r = int(100 + 155 * color_ratio)
            g = int(200 + 55 * color_ratio)
            b = int(255 - 155 * color_ratio)
            
            cv2.line(frame, 
                    (progress_x + i, progress_y),
                    (progress_x + i, progress_y + 30),
                    (b, g, r), 2)
        
        # 进度文本
        progress_text = f"魔力凝聚: {len(recording_frames)}/40 帧"
        text_x = progress_x + 10
        text_y = progress_y - 10
        
        frame = put_cn_safe(frame, progress_text, 
                           (text_x, text_y), 
                           22, (220, 230, 255))
        
        # 动态提示文本
        if progress < 0.3:
            hint_text = "保持手势稳定..."
        elif progress < 0.7:
            hint_text = "魔力凝聚中..."
        else:
            hint_text = "准备释放！"
        
        hint_x = progress_x + progress_width + 20
        hint_y = progress_y + 15
        
        frame = put_cn_safe(frame, hint_text, 
                           (hint_x, hint_y), 
                           20, (255, 220, 100))
        
        # 动态粒子效果
        particle_y = progress_y + 15
        for i in range(int(progress * 10)):
            particle_x = progress_x + random.randint(0, filled_width)
            particle_size = random.randint(3, 6)
            particle_color = (random.randint(200, 255),
                            random.randint(200, 255),
                            random.randint(100, 255))
            
            cv2.circle(frame, (particle_x, particle_y), 
                      particle_size, particle_color, -1)
    
    # ===== 状态指示器 =====
    status_colors = {
        "idle": (120, 200, 255),
        "recording": (255, 220, 0),
        "recognizing": (255, 150, 0),
        "attacking": (255, 50, 50),
        "healing": (50, 255, 100)
    }
    
    current_color = status_colors.get(current_phase, (120, 200, 255))
    status_x = 20
    status_y = 40
    
    # 状态指示器
    cv2.circle(frame, (status_x, status_y), 10, current_color, -1)
    cv2.circle(frame, (status_x, status_y), 10, (255, 255, 255), 2)
    
    # 状态文本
    status_text = f"状态: {current_phase}"
    frame = put_cn_safe(frame, status_text, (status_x + 25, status_y - 5), 
                       18, (255, 255, 255))
    
    return frame

# ======================================================
# 9. 智能录制分析系统
# ======================================================
class SmartRecordingAnalyzer:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.frame_analysis = []
        self.best_gesture = None
        self.best_confidence = 0
        self.confidence_history = []
        
    def analyze_frame(self, hand_landmarks, handedness):
        """分析单帧"""
        gesture_label, confidence, all_probs = self.recognizer.recognize(
            hand_landmarks, handedness, debug=False
        )
        
        analysis = {
            "gesture_label": gesture_label,
            "confidence": confidence,
            "all_probs": all_probs,
            "timestamp": time.time()
        }
        
        self.frame_analysis.append(analysis)
        
        # 更新最佳识别结果
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_gesture = gesture_label
        
        # 记录置信度历史
        self.confidence_history.append(confidence)
        
        return analysis
    
    def get_best_spell(self, magic_system):
        """获取最佳法术"""
        if self.best_gesture and self.best_confidence >= 0.4:
            # 映射到法术
            spell_type = magic_system.label_to_spell.get(self.best_gesture)
            if spell_type:
                return spell_type, self.best_confidence, self.best_gesture
        
        # 如果没有高置信度识别，尝试使用概率最高的
        if self.frame_analysis:
            # 收集所有预测结果
            all_predictions = {}
            for analysis in self.frame_analysis:
                for label, prob in analysis.get("all_probs", []):
                    if label not in all_predictions:
                        all_predictions[label] = []
                    all_predictions[label].append(prob)
            
            # 计算平均概率
            avg_probs = {}
            for label, probs in all_predictions.items():
                avg_probs[label] = sum(probs) / len(probs)
            
            # 选择平均概率最高的
            if avg_probs:
                best_label = max(avg_probs.items(), key=lambda x: x[1])[0]
                avg_confidence = avg_probs[best_label]
                
                if avg_confidence >= 0.3:
                    spell_type = magic_system.label_to_spell.get(best_label)
                    if spell_type:
                        return spell_type, avg_confidence, best_label
        
        return None, 0, None
    
    def reset(self):
        """重置分析器"""
        self.frame_analysis.clear()
        self.best_gesture = None
        self.best_confidence = 0
        self.confidence_history.clear()

# ======================================================
# 10. 增强的MediaPipe双手检测（兼容 mediapipe >= 0.10.x）
# ======================================================

# ---------- 兼容层：把新 Tasks API 包装成旧 solutions 风格 ----------
import urllib.request

def _ensure_hand_model(path="hand_landmarker.task"):
    """自动下载 hand_landmarker.task 模型文件（如果不存在）"""
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print(f"正在下载手部检测模型到 {path} ...")
        try:
            urllib.request.urlretrieve(url, path)
            print("模型下载完成")
        except Exception as e:
            raise RuntimeError(f"模型下载失败: {e}\n请手动下载并放到脚本同目录:\n{url}")
    return path

class _FakeLandmark:
    """把新 API 的 NormalizedLandmark 包装成旧 API 格式（landmark.x/y/z）"""
    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z

class _FakeHandLandmarks:
    """模拟旧 API 的 hand_landmarks 对象"""
    # HAND_CONNECTIONS 与旧版完全相同，直接硬编码
    HAND_CONNECTIONS = frozenset([
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ])
    def __init__(self, landmarks):
        self.landmark = [_FakeLandmark(lm) for lm in landmarks]

class _FakeHandedness:
    """模拟旧 API 的 handedness 对象"""
    class _Cat:
        def __init__(self, label, score):
            self.label = label
            self.score = score
    def __init__(self, category):
        self.classification = [self._Cat(category.category_name, category.score)]

class _FakeResults:
    """模拟旧 API 的 results 对象"""
    def __init__(self, hand_landmarks_list, handedness_list):
        self.multi_hand_landmarks = hand_landmarks_list if hand_landmarks_list else None
        self.multi_handedness   = handedness_list   if handedness_list   else None

class _NewAPIHandsWrapper:
    """用新 Tasks API 实现与旧 mp.solutions.hands.Hands 相同的接口"""
    HAND_CONNECTIONS = _FakeHandLandmarks.HAND_CONNECTIONS

    def __init__(self, max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.3,
                 static_image_mode=False):
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _vision

        model_path = _ensure_hand_model("hand_landmarker.task")
        mode = (_vision.RunningMode.IMAGE
                if static_image_mode
                else _vision.RunningMode.VIDEO)

        options = _vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self._landmarker = _vision.HandLandmarker.create_from_options(options)
        self._running_mode = mode
        self._ts_ms = 0
        from mediapipe.tasks.python.vision import RunningMode as _RM
        self._RM = _RM

    def process(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self._running_mode == self._RM.VIDEO:
            self._ts_ms += 33          # ~30fps
            raw = self._landmarker.detect_for_video(mp_image, self._ts_ms)
        else:
            raw = self._landmarker.detect(mp_image)

        fake_lms  = [_FakeHandLandmarks(h) for h in raw.hand_landmarks]
        fake_heds = [_FakeHandedness(raw.handedness[i][0])
                     for i in range(len(raw.handedness))]
        return _FakeResults(fake_lms, fake_heds)

    def __enter__(self): return self
    def __exit__(self, *a): self._landmarker.close()

class _DrawingUtilsCompat:
    """最小化的 drawing_utils 兼容层"""
    class DrawingSpec:
        def __init__(self, color=(255,255,255), thickness=2, circle_radius=2):
            self.color = color
            self.thickness = thickness
            self.circle_radius = circle_radius

    def draw_landmarks(self, image, hand_landmarks, connections,
                       landmark_drawing_spec=None, connection_drawing_spec=None):
        h, w = image.shape[:2]
        lms = hand_landmarks.landmark

        # 画连线
        conn_spec = connection_drawing_spec or self.DrawingSpec(color=(255,0,0), thickness=2)
        for s, e in (connections or []):
            pt1 = (int(lms[s].x * w), int(lms[s].y * h))
            pt2 = (int(lms[e].x * w), int(lms[e].y * h))
            cv2.line(image, pt1, pt2, conn_spec.color, conn_spec.thickness)

        # 画关键点
        lm_spec = landmark_drawing_spec or self.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
        for lm in lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), lm_spec.circle_radius, lm_spec.color, lm_spec.thickness)

# ---------- 真正的 HandTracker 类 ----------
class ImprovedHandTracker:
    def __init__(self):
        # 统一用兼容包装器，无论 mediapipe 版本
        self.mp_hands = _NewAPIHandsWrapper(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            static_image_mode=False
        )
        # HAND_CONNECTIONS 挂在 mp_hands 上，供外部代码访问
        self.mp_hands.HAND_CONNECTIONS = _FakeHandLandmarks.HAND_CONNECTIONS
        self.mp_draw = _DrawingUtilsCompat()

    def process_frame(self, frame):
        """处理帧并返回手部信息（接口与旧版完全相同）"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        hand_landmarks_list = []
        handedness_list = []

        if results.multi_hand_landmarks:
            hand_landmarks_list = results.multi_hand_landmarks
            if results.multi_handedness:
                handedness_list = results.multi_handedness

            # 绘制手部关键点
            for hand_landmarks in hand_landmarks_list:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )

        return results, hand_landmarks_list, handedness_list, frame
# ======================================================
# 准确率统计模块
# ======================================================
class AccuracyTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_frames = 0
        self.correct_predictions = 0
        self.gesture_stats = {}  # 按手势统计
        self.confidence_history = []
        self.current_session = {
            "start_time": time.time(),
            "gesture_tests": []
        }
    
    def record_prediction(self, predicted_label, true_label=None, confidence=0):
        """记录一次预测结果"""
        self.total_frames += 1
        self.confidence_history.append(confidence)
        
        # 如果有真实标签，可以计算准确率
        if true_label is not None:
            is_correct = (predicted_label == true_label)
            if is_correct:
                self.correct_predictions += 1
            
            # 统计每个手势的准确率
            if true_label not in self.gesture_stats:
                self.gesture_stats[true_label] = {
                    "total": 0,
                    "correct": 0,
                    "confidences": []
                }
            
            self.gesture_stats[true_label]["total"] += 1
            if is_correct:
                self.gesture_stats[true_label]["correct"] += 1
            self.gesture_stats[true_label]["confidences"].append(confidence)
            
            # 记录到当前会话
            self.current_session["gesture_tests"].append({
                "time": time.time(),
                "predicted": predicted_label,
                "true": true_label,
                "confidence": confidence,
                "correct": is_correct
            })
        
        return True
    
    def get_accuracy(self):
        """获取整体准确率"""
        if self.total_frames == 0:
            return 0.0
        return self.correct_predictions / self.total_frames
    
    def get_gesture_accuracy(self, gesture_label):
        """获取特定手势的准确率"""
        if gesture_label not in self.gesture_stats:
            return 0.0
        
        stats = self.gesture_stats[gesture_label]
        if stats["total"] == 0:
            return 0.0
        
        return stats["correct"] / stats["total"]
    
    def get_average_confidence(self):
        """获取平均置信度"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("📊 识别准确率统计")
        print("=" * 60)
        
        print(f"📈 整体统计:")
        print(f"  总测试帧数: {self.total_frames}")
        print(f"  正确识别帧数: {self.correct_predictions}")
        print(f"  整体准确率: {self.get_accuracy()*100:.1f}%")
        print(f"  平均置信度: {self.get_average_confidence():.3f}")
        
        print(f"\n🎯 各手势详细统计:")
        for gesture_label, stats in self.gesture_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            avg_conf = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0
            print(f"  {gesture_label}:")
            print(f"    - 测试次数: {stats['total']}")
            print(f"    - 正确次数: {stats['correct']}")
            print(f"    - 准确率: {accuracy*100:.1f}%")
            print(f"    - 平均置信度: {avg_conf:.3f}")
        
        # 置信度分布
        if self.confidence_history:
            print(f"\n📊 置信度分布:")
            conf_ranges = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
            for low, high in conf_ranges:
                count = sum(1 for c in self.confidence_history if low <= c < high)
                percentage = count / len(self.confidence_history) * 100
                print(f"  {low:.1f}-{high:.1f}: {count}帧 ({percentage:.1f}%)")
        
        print("=" * 60)
# ======================================================
# 11. 主程序
# ======================================================
def main():
    print("=" * 60)
    print("创意魔法手势识别系统")
    print("=" * 60)
    
    # 初始化系统
    recognizer = EnhancedGestureRecognizer("trained_models\\latest_sign_language_model.joblib")
    if recognizer.model is None:
        print("无法加载模型，程序退出")
        return
    
    # 初始化敌人
    enemy = Enemy("暗影巫师", 200)
    
    # 初始化魔法系统
    magic_system = EnhancedMagicSystem(enemy)
    
    # 初始化智能录制分析器
    analyzer = SmartRecordingAnalyzer(recognizer)
    
    # 初始化改进的手部跟踪器
    hand_tracker = ImprovedHandTracker()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 状态变量
    state = "idle"
    recording = False
    recording_frames = []
    current_spell = None
    spell_effect = None
    attack_phase = False
    attack_result = None
    attack_result_display_time = 0
    last_frame_time = time.time()
    
    # 手部检测状态
    hands_detected = 0
    last_hands_detection = time.time()
    
    # FPS计算
    fps = 0
    last_fps_time = time.time()
    fps_counter = 0
    
    print("\n系统初始化完成")
    print(f"敌人: {enemy.name} (生命值: {enemy.health}/{enemy.max_health})")
    print("控制键: S=开始, E=结束/攻击, M=魔力, R=重置, D=调试, ESC=退出")
    print("手部检测: 保持双手分开以获得最佳识别效果")
    print("法术效果:")
    print("  - 庆祝咒: 造成20伤害")
    print("  - 护盾术: 造成15伤害")
    print("  - 闪电术: 造成30伤害")
    print("  - 治疗术: 恢复40生命")
    print("\n系统启动中...")
    
    # 主循环
    while True:
        try:
            # 计算时间差
            current_time = time.time()
            delta_time = current_time - last_frame_time
            last_frame_time = current_time
            
            # 计算FPS
            fps_counter += 1
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_fps_time = current_time
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break
            
            # 镜像翻转
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # 使用改进的手部跟踪器
            results, hand_landmarks_list, handedness_list, frame = hand_tracker.process_frame(frame)
            
            # 实时检测手势
            current_gesture_label = None
            current_confidence = 0
            
            if len(hand_landmarks_list) >= 2:
                gesture_label, confidence, _ = recognizer.recognize(
                    hand_landmarks_list,
                    handedness_list,
                    debug=False
                )
                if gesture_label:
                    current_gesture_label = gesture_label
                    current_confidence = confidence
                    
                # 更新手部检测状态
                hands_detected = len(hand_landmarks_list)
                last_hands_detection = current_time
            else:
                # 如果没有检测到双手，在顶部居中显示提示
                if current_time - last_hands_detection > 2.0:
                    tips_y = 50  # 移动到接近顶部的位置
                    tips_width = 400
                    tips_height = 50
                    tips_x = w // 2 - tips_width // 2
                    
                    # 提示背景
                    cv2.rectangle(frame, (tips_x, tips_y), 
                                 (tips_x + tips_width, tips_y + tips_height),
                                 (30, 25, 40), -1)
                    cv2.rectangle(frame, (tips_x, tips_y), 
                                 (tips_x + tips_width, tips_y + tips_height),
                                 (100, 150, 200), 2)
                    
                    # 居中显示文本
                    tip_text = "请确保双手都在摄像头范围内"
                    text_width = len(tip_text) * 12  # 估算文本宽度
                    text_x = tips_x + (tips_width - 1.8*text_width) // 2
                    
                    frame = put_cn_safe(frame, tip_text,
                                       (text_x, tips_y + 15),
                                       22, (255, 220, 100))
            
            # ===== 录制逻辑 =====
            if recording and len(hand_landmarks_list) >= 2:
                recording_frames.append({
                    "hand_landmarks": hand_landmarks_list,
                    "handedness": handedness_list,
                    "timestamp": current_time
                })
                
                # 实时分析帧
                analyzer.analyze_frame(
                    hand_landmarks_list,
                    handedness_list
                )
                
                # 自动结束录制（录够40帧）
                if len(recording_frames) >= 40:
                    print("录制帧数已满，准备释放法术...")
                    recording = False
                    state = "attacking"
                    attack_phase = True
            
            # ===== 攻击阶段 =====
            if attack_phase and not recording:
                # 获取最佳法术
                best_spell, confidence, gesture_label = analyzer.get_best_spell(magic_system)
                
                if best_spell:
                    print(f"\n智能分析完成:")
                    print(f"  最佳手势: {gesture_label}")
                    print(f"  置信度: {confidence:.3f}")
                    print(f"  对应法术: {SPELL_CONFIG.get(best_spell, {}).get('name', '未知')}")
                    
                    # 施放法术
                    success, message, damage, healing = magic_system.cast_spell(best_spell)
                    
                    if success:
                        current_spell = best_spell
                        
                        # 创建特效
                        spell_effect = EnhancedSpellEffect(
                            best_spell,
                            (w // 2, h // 2),
                            confidence,
                            w, h
                        )
                        
                        # 设置攻击结果
                        attack_result = {
                            "spell": best_spell,
                            "damage": damage,
                            "healing": healing,
                            "confidence": confidence,
                            "message": message
                        }
                        attack_result_display_time = current_time + 3.0
                        
                        print(f"{message}")
                        if damage > 0:
                            print(f"造成伤害: {damage}")
                        if healing > 0:
                            print(f"治疗效果: {healing}")
                    else:
                        attack_result = {
                            "spell": None,
                            "damage": 0,
                            "healing": 0,
                            "confidence": 0,
                            "message": message
                        }
                        attack_result_display_time = current_time + 2.0
                        print(f"{message}")
                
                # 重置分析器
                analyzer.reset()
                recording_frames.clear()
                attack_phase = False
                state = "idle"
            
            # ===== 更新系统 =====
            magic_system.update(delta_time)
            enemy.update(delta_time)
            
            # ===== 检查攻击结果显示时间 =====
            if attack_result and current_time > attack_result_display_time:
                attack_result = None
            
            # ===== 绘制敌人 =====
            frame = enemy.draw(frame, w, h)
            
            # ===== 绘制UI =====
            frame = draw_enhanced_magic_ui(frame, magic_system, w, h,
                                         current_gesture_label, current_confidence,
                                         recording, recording_frames, state)
            
            # ===== 法术特效 =====
            if spell_effect:
                if not spell_effect.update(frame):
                    spell_effect = None
                    state = "idle"
            
            # ===== 显示攻击结果 =====
            if attack_result and current_time <= attack_result_display_time:
                result_y = h // 2 + 150
                result_panel_height = 80
                
                # 结果面板
                cv2.rectangle(frame, (w//2 - 220, result_y), 
                             (w//2 + 220, result_y + result_panel_height),
                             (40, 35, 50), -1)
                cv2.rectangle(frame, (w//2 - 220, result_y), 
                             (w//2 + 220, result_y + result_panel_height),
                             (120, 100, 150), 2)
                
                # 结果文本
                if attack_result["spell"]:
                    spell_name = SPELL_CONFIG.get(attack_result["spell"], {}).get("name", "法术")
                    result_text = f"{spell_name} 释放成功！"
                    
                    if attack_result["damage"] > 0:
                        result_text += f"  造成 {attack_result['damage']} 点伤害"
                    if attack_result["healing"] > 0:
                        result_text += f"  治疗 {attack_result['healing']} 点生命"
                else:
                    result_text = attack_result.get("message", "施法失败")
                
                frame = put_cn_safe(frame, result_text,
                                   (w//2 - 200, result_y + 35),
                                   22, (255, 255, 200))
            
            # ===== 显示敌人状态信息 =====
            if enemy.status == "defeated":
                victory_y = 100
                victory_width = 320
                victory_x = w // 2 - victory_width // 2
                
                # 胜利背景
                cv2.rectangle(frame, (victory_x, victory_y), 
                             (victory_x + victory_width, victory_y + 60),
                             (50, 40, 60), -1)
                cv2.rectangle(frame, (victory_x, victory_y), 
                             (victory_x + victory_width, victory_y + 60),
                             (200, 180, 100), 2)
                
                victory_text = f"敌人已被击败！按 R 键重置"
                frame = put_cn_safe(frame, victory_text,
                                   (victory_x + 20, victory_y + 35),
                                   24, (255, 240, 180))
            
            # ===== 显示系统信息（移除右上角绿色状态）=====
            # 只显示FPS
            fps_text = f"FPS: {fps}"
            cv2.putText(frame, fps_text, (w - 120, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            
            # ===== 显示窗口 =====
            cv2.imshow("Sign Language", frame)
            
            # ===== 键盘控制 =====
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                if state == "idle" and hands_detected >= 2:
                    recording = True
                    state = "recording"
                    analyzer.reset()
                    recording_frames.clear()
                    sound.play("ui_start")
                    print("\n" + "=" * 50)
                    print("开始智能录制...")
                    print("系统将实时分析手势，结束时自动选择最佳法术")
                elif hands_detected < 2:
                    print("请确保检测到两只手再开始录制")
            
            elif key == ord('e') or key == ord('E'):
                if state == "recording":
                    recording = False
                    state = "attacking"
                    attack_phase = True
                    sound.play("ui_end")
                    print("\n录制结束，开始智能分析...")
            
            elif key == ord('m') or key == ord('M'):
                magic_system.mana = magic_system.max_mana
                sound.play("ui_success")
                print("魔力已恢复")
            
            elif key == ord('r') or key == ord('R'):
                if magic_system.combo > 0:
                    magic_system.combo = 0
                    print("连击已重置")
                elif enemy.status == "defeated":
                    # 重置敌人
                    enemy.health = enemy.max_health
                    enemy.status = "alive"
                    enemy.hurt_effect = 0
                    enemy.heal_effect = 0
                    print(f"敌人已重置，生命值: {enemy.health}/{enemy.max_health}")
            
            elif key == ord('d') or key == ord('D'):
                if len(hand_landmarks_list) >= 2:
                    print("\n" + "=" * 60)
                    print("调试模式: 实时识别当前帧")
                    gesture_label, confidence, all_probs = recognizer.recognize(
                        hand_landmarks_list,
                        handedness_list,
                        debug=True
                    )
                    if gesture_label:
                        print(f"识别结果: {gesture_label} (置信度: {confidence:.3f})")
                    else:
                        print(f"未识别 (置信度: {confidence:.3f})")
                    print(f"敌人状态: {enemy.status}, 生命值: {enemy.health}/{enemy.max_health}")
                    print(f"玩家魔力: {int(magic_system.mana)}/{int(magic_system.max_mana)}")
            
            elif key == 27:  # ESC
                print("\n退出系统...")
                break
        
        except Exception as e:
            print(f"处理帧时出错: {e}")
            import traceback
            traceback.print_exc()
            # 重置状态，继续运行
            state = "idle"
            recording = False
            attack_phase = False
            continue
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    print("系统已关闭")

# ======================================================
# 程序入口
# ======================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()