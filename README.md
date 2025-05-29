### 🖌️ 手写数字识别画板
这是一个基于 Python 的交互式手写数字识别应用，用户可以在画板上手写数字，系统将通过以下三种算法进行识别：

K-近邻算法（KNN）：利用 OpenCV 实现，适合初学者理解和使用。

支持向量机（SVM）：通过 OpenCV 实现，具有较高的准确率。

卷积神经网络（CNN）：使用 TensorFlow/Keras 构建，适合处理图像数据，识别效果更佳。

该项目旨在帮助用户了解不同算法在手写数字识别中的应用和效果对比。

🚀 快速开始
🧪 创建和激活虚拟环境
为避免依赖冲突，建议为项目创建一个虚拟环境：

bash
复制
编辑
# 创建虚拟环境

```bash
python3.8 -m venv venv
```
# 激活虚拟环境（Windows）
```
venv\Scripts\activate
```
# 激活虚拟环境（macOS/Linux）
```
source venv/bin/activate
```
📦 安装依赖
在激活的虚拟环境中，使用以下命令安装项目依赖：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
🏃‍♂️ 运行应用
在项目根目录下，运行以下命令启动应用：

```bash
python write.py
```
📝 使用说明
在画板上手写数字。

选择使用的算法（KNN、SVM 或 CNN）。

点击“识别”按钮，查看识别结果。

🗂️ 项目结构
```bash
handwriting-recognition-board/
├── gui.py                  # 主程序文件
├── model/                  # 存放模型文件
│   ├── model_structure.json
│   ├── model_weight.h5
│   ├── svm_data.dat
│   └── knn_data_train.npz
├── IMG/
│   └── sum/
│       └── 1.jpg           # 示例图片
├── new.jpg                 # 保存的手写数字图片
├── requirements.txt        # 项目依赖列表
└── README.md               # 项目说明文件
```

📷 示例截图

![image](https://github.com/user-attachments/assets/bfa97145-daed-4a28-aa45-f03817729b7d)


## 有问题欢迎发邮箱问我，欢迎参考代码。

