import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ===================== 路径配置 =====================
DATA_PATH = "/Users/gejiangyang/Desktop/CNN人脸识别/face_data"
TEST_IMG_PATH = "/Users/gejiangyang/Desktop/CNN人脸识别/ceshi17.jpg"

MODEL_PATH = "siamese_face_cnn.pth"
PROTOTYPE_PATH = "face_prototypes.pth"

BATCH_SIZE = 16
EPOCHS = 20
LR = 5e-4

# 距离阈值：越小越严格
# 如果经常误识别，调小，例如 0.65
# 如果经常不确定，调大，例如 0.90
DISTANCE_THRESHOLD = 0.80


# ===================== 设备选择 =====================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ===================== 小型 CNN 特征提取网络 =====================
class SmallFaceEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SmallFaceEmbeddingCNN, self).__init__()

        self.features = nn.Sequential(
            # 3 x 112 x 112 -> 32 x 56 x 56
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 32 x 56 x 56 -> 64 x 28 x 28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 64 x 28 x 28 -> 128 x 14 x 14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 128 x 14 x 14 -> 256 x 7 x 7
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.embedding(x)

        # 归一化，方便计算欧氏距离
        x = F.normalize(x, p=2, dim=1)
        return x


# ===================== Contrastive Loss =====================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        """
        label = 1 表示同一个人
        label = 0 表示不同人
        """
        distance = F.pairwise_distance(feat1, feat2)

        positive_loss = label * torch.pow(distance, 2)
        negative_loss = (1 - label) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0),
            2
        )

        loss = torch.mean(positive_loss + negative_loss)
        return loss


# ===================== 图片读取工具 =====================
def load_image_paths(root_dir):
    person_names = sorted([
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ])

    person_to_images = {}

    for person in person_names:
        folder = os.path.join(root_dir, person)

        imgs = (
            glob.glob(os.path.join(folder, "*.jpg")) +
            glob.glob(os.path.join(folder, "*.jpeg")) +
            glob.glob(os.path.join(folder, "*.png")) +
            glob.glob(os.path.join(folder, "*.bmp"))
        )

        person_to_images[person] = imgs

    return person_names, person_to_images


# ===================== Siamese 数据集 =====================
class SiameseFaceDataset(Dataset):
    def __init__(self, root_dir, pairs_per_epoch=1000):
        self.root_dir = root_dir
        self.pairs_per_epoch = pairs_per_epoch

        self.person_names, self.person_to_images = load_image_paths(root_dir)

        # 过滤掉照片少于2张的人
        self.person_names = [
            p for p in self.person_names
            if len(self.person_to_images[p]) >= 2
        ]

        if len(self.person_names) < 2:
            raise ValueError("至少需要2个人，并且每个人至少2张照片。")

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomCrop((112, 112)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.2
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        # 50% 正样本，50% 负样本
        same_person = random.random() < 0.5

        if same_person:
            person = random.choice(self.person_names)
            img1_path, img2_path = random.sample(
                self.person_to_images[person],
                2
            )
            label = 1.0

        else:
            person1, person2 = random.sample(self.person_names, 2)

            img1_path = random.choice(self.person_to_images[person1])
            img2_path = random.choice(self.person_to_images[person2])

            label = 0.0

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        label = torch.tensor(label, dtype=torch.float32)

        return img1, img2, label


# ===================== 训练 Siamese CNN =====================
def train_siamese_model():
    device = get_device()
    print("当前设备：", device)

    dataset = SiameseFaceDataset(
        DATA_PATH,
        pairs_per_epoch=1200
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = SmallFaceEmbeddingCNN(embedding_dim=128).to(device)

    criterion = ContrastiveLoss(margin=1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.5
    )

    print("\n开始训练 Siamese CNN...\n")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for img1, img2, label in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            feat1 = model(img1)
            feat2 = model(img2)

            loss = criterion(feat1, feat2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)

        print(
            f"轮次 {epoch + 1:3d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print("\n✅ Siamese CNN 训练完成")
    print("模型已保存：", MODEL_PATH)

    return model


# ===================== 验证/测试用 transform =====================
def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    ])


# ===================== 构建每个人的平均特征 =====================
def build_prototypes(model):
    device = get_device()
    model.eval()

    person_names, person_to_images = load_image_paths(DATA_PATH)
    transform = get_eval_transform()

    prototypes = {}

    print("\n开始构建每个人的平均人脸特征...\n")

    with torch.no_grad():
        for person in person_names:
            features = []

            for img_path in person_to_images[person]:
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)

                feat = model(img)
                features.append(feat.cpu())

            if len(features) == 0:
                continue

            features = torch.cat(features, dim=0)

            # 该人物所有照片特征取平均
            prototype = torch.mean(features, dim=0, keepdim=True)

            # 再归一化一次
            prototype = F.normalize(prototype, p=2, dim=1)

            prototypes[person] = prototype

            print(f"{person}: 使用 {len(features)} 张图片构建原型")

    torch.save({
        "prototypes": prototypes,
        "person_names": person_names
    }, PROTOTYPE_PATH)

    print("\n✅ 人脸原型库已保存：", PROTOTYPE_PATH)


# ===================== 加载模型 =====================
def load_model():
    device = get_device()

    model = SmallFaceEmbeddingCNN(embedding_dim=128).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    return model


# ===================== 预测单张图片 =====================
def predict_face(model, img_path):
    device = get_device()

    checkpoint = torch.load(PROTOTYPE_PATH, map_location="cpu")
    prototypes = checkpoint["prototypes"]

    transform = get_eval_transform()

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        test_feat = model(img).cpu()

    best_name = None
    best_distance = float("inf")

    print("\n各人物距离：")

    for person, prototype in prototypes.items():
        distance = F.pairwise_distance(test_feat, prototype).item()

        print(f"{person}: {distance:.4f}")

        if distance < best_distance:
            best_distance = distance
            best_name = person

    if best_distance > DISTANCE_THRESHOLD:
        return {
            "result": "不确定",
            "best_name": best_name,
            "distance": best_distance
        }

    return {
        "result": best_name,
        "best_name": best_name,
        "distance": best_distance
    }


# ===================== 主程序 =====================
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(PROTOTYPE_PATH):
        print("✅ 检测到已训练模型和原型库，直接加载测试...")
        model = load_model()
    else:
        print("⚠️ 未检测到模型或原型库，开始训练...")
        model = train_siamese_model()
        build_prototypes(model)

    result = predict_face(model, TEST_IMG_PATH)

    print("\n🔍 预测结果：")
    print("识别结果：", result["result"])
    print("最接近人物：", result["best_name"])
    print("特征距离：", round(result["distance"], 4))