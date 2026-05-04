import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

# ===================== 1. 分类专用CNN模型 =====================
class FaceClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceClassifierCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # 全连接层 → 直接输出类别概率
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)  # 防止过拟合
        self.fc2 = nn.Linear(512, num_classes)  # 输出：类别数

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # 直接输出分类分数
        return x

# ===================== 2. 自动读取人脸数据集（不变） =====================
class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        
        self.person_names = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        
        for idx, name in enumerate(self.person_names):
            folder = os.path.join(root_dir, name)
            imgs = glob.glob(folder + "/*.jpg") + glob.glob(folder + "/*.png")
            self.image_paths.extend(imgs)
            self.labels.extend([idx] * len(imgs))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ===================== 3. 训练函数（已修复） =====================
def train_face_model():
    device = torch.device("mps")
    
    # 数据集路径
    data_path = "/Users/gejiangyang/Desktop/CNN人脸识别/face_data"
    dataset = FaceDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 自动获取类别数量（几个人）
    num_classes = len(dataset.person_names)
    model = FaceClassifierCNN(num_classes=num_classes).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("开始训练人脸分类模型...\n")
    
    # 训练30轮
    for epoch in range(40):
        model.train()
        loss_sum = 0
        correct = 0
        total = 0
        
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            outputs = model(img)
            loss = criterion(outputs, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        
        acc = 100 * correct / total
        print(f"轮次 {epoch+1:2d} | 损失: {loss_sum:.2f} | 准确率: {acc:.1f}%")

    torch.save(model.state_dict(), "face_trained_model.pth")
    print("\n✅ 训练完成！模型已保存")
    return model

# ===================== 4. 预测函数 =====================
def predict_face(model, img_path):
    device = torch.device("mps")
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # 增加batch维度
    
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        pred_idx = torch.argmax(outputs).item()
    
    # 读取姓名
    data_path = "/Users/gejiangyang/Desktop/CNN人脸识别/face_data"
    person_names = sorted(os.listdir(data_path))
    return person_names[pred_idx]

# ===================== 主程序 =====================
if __name__ == "__main__":
    device = torch.device("mps")
    data_path = "/Users/gejiangyang/Desktop/CNN人脸识别/face_data"
    weight_path = "face_trained_model.pth"  # 权重文件路径
    
    # 自动获取人数
    person_names = sorted(os.listdir(data_path))
    num_classes = len(person_names)
    
    # ==============================================
    # 核心逻辑：检测权重文件是否存在
    # ==============================================
    if os.path.exists(weight_path):
        # ✅ 有权重 → 直接加载模型，不训练！
        print("✅ 检测到已训练模型，直接加载权重进行测试...")
        model = FaceClassifierCNN(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(weight_path))
    else:
        # ❌ 无权重 → 从头训练
        print("⚠️ 未检测到模型权重，开始重新训练...")
        model = train_face_model()

    # 测试图片（你可以随时换路径）
    test_img_path = "/Users/gejiangyang/Desktop/CNN人脸识别/ceshi10.jpg"
    result = predict_face(model, test_img_path)
    print(f"\n🔍 预测结果：这是 -> {result}")