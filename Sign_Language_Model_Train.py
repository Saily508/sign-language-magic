# train_complete_model_chinese.py
import numpy as np
import pickle
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
def setup_chinese_font():
    """设置matplotlib支持中文显示"""
    import platform
    import matplotlib
    
    system = platform.system()
    
    if system == "Windows":
        # Windows系统
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                matplotlib.font_manager.fontManager.addfont(font_path)
                font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                plt.rcParams['font.sans-serif'] = [font_name]
                break
    
    elif system == "Darwin":  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang HK', 'STHeiti']
    
    elif system == "Linux":
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

class CompleteSignLanguageTrainer:
    def __init__(self, model_type='random_forest'):
        """
        初始化完整手语训练器
        
        Args:
            model_type: 模型类型 ('random_forest', 'svm', 'knn', 'mlp', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.classes_ = None
        self.sign_info = None
        
        # 设置中文字体
        setup_chinese_font()
        
        print("=" * 70)
        print("完整手语识别模型训练系统")
        print("=" * 70)
    
    def load_dataset(self, data_path=None):
        """
        加载手语数据集
        
        Args:
            data_path: 数据文件路径，如果为None则自动查找最新文件
            
        Returns:
            success: 是否成功加载
        """
        data_dir = "complete_sign_data"
        
        if not os.path.exists(data_dir):
            print(f"错误: 数据目录 '{data_dir}' 不存在")
            print("请先运行数据收集脚本")
            return False
        
        # 如果未指定路径，查找最新文件
        if data_path is None:
            data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
            if not data_files:
                print("错误: 未找到数据文件")
                return False
            
            # 选择最新文件
            data_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
            data_path = os.path.join(data_dir, data_files[0])
            print(f"自动选择最新文件: {data_files[0]}")
        
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.features = np.array(data['features'])
            self.labels = np.array(data['labels'])
            self.sign_info = data.get('sign_language_info', {})
            self.metadata = data.get('metadata', {})
            
            print(f"\n✅ 数据加载成功: {os.path.basename(data_path)}")
            print(f"📊 总样本数: {self.features.shape[0]}")
            print(f"🔢 特征维度: {self.features.shape[1]}")
            
            if self.metadata:
                print(f"📅 收集日期: {self.metadata.get('collection_date', '未知')}")
                print(f"⏱️  样本频率: {self.metadata.get('sample_rate', '未知')}")
            
            # 显示手语信息
            if self.sign_info:
                print(f"\n📋 包含的手语 ({len(self.sign_info)} 种):")
                for key, info in self.sign_info.items():
                    count = np.sum(self.labels == key)
                    print(f"  {info['chinese']} ({info['english']}): {count} 个样本")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def preprocess_data(self, test_size=0.25):
        """
        数据预处理
        
        Args:
            test_size: 测试集比例
            
        Returns:
            success: 是否成功
        """
        if not hasattr(self, 'features'):
            print("错误: 请先加载数据")
            return False
        
        # 编码标签
        self.y_encoded = self.label_encoder.fit_transform(self.labels)
        self.classes_ = self.label_encoder.classes_
        
        # 标准化特征
        self.X_scaled = self.scaler.fit_transform(self.features)
        
        # 划分训练测试集（分层抽样）
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y_encoded,
            test_size=test_size,
            random_state=42,
            stratify=self.y_encoded
        )
        
        print(f"\n✅ 数据预处理完成")
        print(f"📚 训练集: {self.X_train.shape[0]} 个样本")
        print(f"🧪 测试集: {self.X_test.shape[0]} 个样本")
        print(f"🎯 类别数: {len(self.classes_)}")
        
        return True
    
    def create_model(self):
        """创建和配置模型"""
        print(f"\n🤖 创建 {self.model_type} 模型...")
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                verbose=1
            )
            
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            
        elif self.model_type == 'knn':
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                algorithm='auto',
                n_jobs=-1
            )
            
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                verbose=True
            )
            
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=1
            )
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        print(f"✅ {self.model_type} 模型创建成功")
        return True
    
    def train_model(self):
        """训练模型"""
        print(f"\n🚀 开始训练模型...")
        
        # 交叉验证评估
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                   cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"📊 交叉验证结果:")
        print(f"   平均准确率: {cv_scores.mean():.4f}")
        print(f"   标准差: {cv_scores.std():.4f}")
        print(f"   各折准确率: {cv_scores}")
        
        # 训练最终模型
        print(f"\n🎯 训练最终模型...")
        self.model.fit(self.X_train, self.y_train)
        print(f"✅ 模型训练完成")
        
        return True
    
    def evaluate_model(self):
        """评估模型性能"""
        print(f"\n📈 模型性能评估")
        print("=" * 50)
        
        # 预测
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)
        
        # 准确率
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"🏆 测试集准确率: {accuracy:.4f}")
        
        # 平均置信度
        avg_confidence = np.max(y_prob, axis=1).mean()
        print(f"🎯 平均置信度: {avg_confidence:.4f}")
        
        # 分类报告
        print(f"\n📋 详细分类报告:")
        target_names = [self.sign_info.get(cls, {'chinese': cls}).get('chinese', cls) 
                       for cls in self.classes_]
        
        report = classification_report(self.y_test, y_pred, 
                                     target_names=target_names,
                                     digits=4)
        print(report)
        
        # 混淆矩阵
        self.plot_confusion_matrix(y_pred)
        
        # 特征重要性（如果可用）
        if hasattr(self.model, 'feature_importances_'):
            self.plot_feature_importance()
        
        return accuracy
    
    def plot_confusion_matrix(self, y_pred):
        """绘制混淆矩阵（中文）"""
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # 使用热图显示混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   cbar_kws={'label': '样本数量'},
                   square=True)
        
        # 设置中文标签
        labels = [self.sign_info.get(cls, {'chinese': cls}).get('chinese', cls) 
                 for cls in self.classes_]
        
        plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)
        
        plt.title('混淆矩阵 - 手语识别模型', fontsize=16, pad=20)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'results/confusion_matrix_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 混淆矩阵已保存到: {save_path}")
        plt.show()
    
    def plot_feature_importance(self):
        """绘制特征重要性（如果模型支持）"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # 获取最重要的20个特征
            indices = np.argsort(importances)[::-1][:20]
            
            plt.figure(figsize=(12, 6))
            plt.title('Top 20 特征重要性', fontsize=16)
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), indices, rotation=45)
            plt.xlabel('特征索引', fontsize=12)
            plt.ylabel('重要性', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'results/feature_importance_{timestamp}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 特征重要性图已保存到: {save_path}")
            plt.show()
    
    def save_model(self, model_dir="trained_models"):
        """
        保存训练好的模型
        
        Args:
            model_dir: 模型保存目录
            
        Returns:
            model_path: 模型保存路径
        """
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"complete_sign_language_model_{self.model_type}_{timestamp}.joblib"
        model_path = os.path.join(model_dir, model_name)
        
        # 保存模型数据
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'classes': self.classes_,
            'sign_info': self.sign_info,
            'metadata': self.metadata,
            'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'feature_dimension': self.X_train.shape[1],
            'description': '完整手语识别模型 - 支持庆祝术、护盾术、治疗术、闪电术'
        }
        
        joblib.dump(model_data, model_path)
        
        # 保存为最新版本
        latest_path = os.path.join(model_dir, "latest_sign_language_model.joblib")
        joblib.dump(model_data, latest_path)
        
        # 保存模型信息文件
        info_path = os.path.join(model_dir, "model_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("完整手语识别模型信息\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"模型类型: {self.model_type}\n")
            f.write(f"训练时间: {model_data['training_time']}\n")
            f.write(f"特征维度: {model_data['feature_dimension']}\n")
            f.write(f"训练样本: {self.X_train.shape[0]}\n")
            f.write(f"测试样本: {self.X_test.shape[0]}\n")
            f.write(f"\n支持的手语:\n")
            
            for cls in self.classes_:
                if cls in self.sign_info:
                    sign_name = self.sign_info[cls]['chinese']
                    sign_desc = self.sign_info[cls]['sign_description']
                    f.write(f"  {sign_name}: {sign_desc}\n")
            
            f.write(f"\n模型文件: {model_name}\n")
            f.write(f"最新版本: latest_sign_language_model.joblib\n")
        
        print(f"\n💾 模型保存成功!")
        print(f"📁 模型文件: {model_path}")
        print(f"📁 最新版本: {latest_path}")
        print(f"📄 模型信息: {info_path}")
        
        return model_path
    
    def run_complete_training(self, data_path=None):
        """
        运行完整的训练流程
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            accuracy: 模型准确率
        """
        print("\n" + "=" * 70)
        print("开始完整训练流程")
        print("=" * 70)
        
        # 1. 加载数据
        if not self.load_dataset(data_path):
            return None
        
        # 2. 预处理数据
        if not self.preprocess_data():
            return None
        
        # 3. 创建模型
        if not self.create_model():
            return None
        
        # 4. 训练模型
        if not self.train_model():
            return None
        
        # 5. 评估模型
        accuracy = self.evaluate_model()
        
        # 6. 保存模型
        if accuracy > 0.75:  # 准确率超过75%才保存
            self.save_model()
        else:
            print(f"\n⚠️  模型准确率较低 ({accuracy:.4f})，建议:")
            print("  1. 收集更多训练数据")
            print("  2. 尝试不同的模型类型")
            print("  3. 检查数据质量")
        
        return accuracy

def select_model_type():
    """选择模型类型"""
    print("\n请选择模型类型:")
    print("1. Random Forest (随机森林) - 推荐，鲁棒性强")
    print("2. SVM (支持向量机) - 适合小样本")
    print("3. KNN (K近邻) - 简单快速")
    print("4. MLP (神经网络) - 适合复杂模式")
    print("5. Gradient Boosting (梯度提升) - 高准确率")
    
    choices = {'1': 'random_forest', '2': 'svm', '3': 'knn', 
               '4': 'mlp', '5': 'gradient_boosting'}
    
    while True:
        choice = input("\n请输入编号 (1-5, 默认1): ").strip()
        if choice == '':
            return 'random_forest'
        elif choice in choices:
            return choices[choice]
        else:
            print("无效选择，请重新输入")

def select_dataset():
    """选择数据集文件"""
    data_dir = "complete_sign_data"
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 不存在")
        return None
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if not data_files:
        print("错误: 未找到数据文件")
        return None
    
    print(f"\n找到 {len(data_files)} 个数据文件:")
    for i, f in enumerate(data_files, 1):
        file_path = os.path.join(data_dir, f)
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"{i}. {f} (修改时间: {file_time.strftime('%Y-%m-%d %H:%M')})")
    
    print(f"{len(data_files) + 1}. 自动选择最新文件")
    
    while True:
        try:
            choice = input(f"\n请选择文件编号 (1-{len(data_files)+1}, 默认{len(data_files)+1}): ").strip()
            
            if choice == '':
                # 自动选择最新文件
                data_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
                return os.path.join(data_dir, data_files[0])
            
            choice = int(choice)
            if choice == len(data_files) + 1:
                # 自动选择最新文件
                data_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
                return os.path.join(data_dir, data_files[0])
            elif 1 <= choice <= len(data_files):
                return os.path.join(data_dir, data_files[choice - 1])
            else:
                print(f"无效选择，请输入 1-{len(data_files)+1} 之间的数字")
                
        except ValueError:
            print("请输入有效的数字")

def main():
    """主函数"""
    print("=" * 70)
    print("完整手语识别模型训练系统")
    print("=" * 70)
    
    # 1. 选择数据集
    print("\n步骤 1: 选择数据集")
    data_path = select_dataset()
    if not data_path:
        return
    
    # 2. 选择模型类型
    print("\n步骤 2: 选择模型类型")
    model_type = select_model_type()
    
    # 3. 创建训练器
    trainer = CompleteSignLanguageTrainer(model_type=model_type)
    
    # 4. 运行训练
    print(f"\n开始训练 {model_type} 模型...")
    accuracy = trainer.run_complete_training(data_path)
    
    if accuracy is not None:
        print(f"\n🎉 训练流程完成!")
        print(f"📊 最终准确率: {accuracy:.4f}")
        
        if accuracy > 0.85:
            print("✅ 模型性能优秀!")
        elif accuracy > 0.75:
            print("👍 模型性能良好")
        else:
            print("⚠️  模型需要改进")
    
    print("\n" + "=" * 70)
    print("训练系统结束")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n按Enter键退出程序...")