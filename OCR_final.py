import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import csv

torch.set_num_threads(2)

# 训练标签文件路径
TRAIN_LABEL_PATH = r'C:\Users\全天悦\PycharmProjects\PythonProject\机器学习\train_list.txt'
# 训练图片文件夹
TRAIN_IMAGE_DIR = r'D:\学习\test\OCR零件\image'

# 【测试/评估】标准答案 CSV 路径
TEST_CSV_PATH = r'D:\学习\test\OCR零件\labels1127.csv'
# 【测试/评估】测试图片所在文件夹
TEST_IMAGE_DIR = r'D:\学习\test\OCR零件\image'

# 模型保存路径
MODEL_SAVE_PATH = 'iron_ocr_best.pth'


CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
char2idx = {char: i + 1 for i, char in enumerate(CHAR_SET)}
idx2char = {i + 1: char for i, char in enumerate(CHAR_SET)}
idx2char[0] = '-'
NUM_CLASSES = len(CHAR_SET) + 1


class IronDataset(Dataset):
    def __init__(self, label_path, image_dir, transform=None):
        self.data = []
        self.transform = transform
        self.image_dir = image_dir

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"找不到标签文件: {label_path}，请检查路径！")

        ext = os.path.splitext(label_path)[1].lower()

        if ext == '.csv':
            with open(label_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader, None)  # 跳过表头
                for row in reader:
                    if len(row) < 2:
                        continue
                    img_name = row[0].strip()
                    label = row[1].strip()
                    img_path = os.path.join(self.image_dir, img_name)
                    if os.path.exists(img_path):
                        self.data.append((img_path, label))
                    else:
                        print(f"警告: 找不到图片文件 -> {img_path}")
        else:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                line = line.replace('\\', '/')
                parts = line.split()
                if len(parts) < 2:
                    continue

                img_name = os.path.basename(parts[0])
                label = " ".join(parts[1:])

                img_path = os.path.join(self.image_dir, img_name)

                if os.path.exists(img_path):
                    self.data.append((img_path, label))
                else:
                    print(f"警告: 找不到图片文件 -> {img_path}")

        print(f"成功加载数据: {len(self.data)} 条")

        if len(self.data) == 0:
            raise ValueError(("没有成功加载任何数据！\n"
                              "1. 请检查标签文件里的内容格式。\n"
                              "2. 请确认图片文件夹路径是否正确。"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_attempts = len(self.data)
        attempts = 0

        while attempts < max_attempts:
            path, label = self.data[idx]
            try:
                img = Image.open(path).convert('L')
                if self.transform:
                    img = self.transform(img)

                target = [char2idx[c] for c in label if c in char2idx]
                if len(target) == 0:
                    idx = (idx + 1) % len(self.data)
                    attempts += 1
                    continue

                return img, torch.IntTensor(target), len(target)

            except Exception as e:
                print(f"读取图片失败 {path}: {e}")
                idx = (idx + 1) % len(self.data)
                attempts += 1

        dummy_img = torch.zeros(1, 32, 256)
        return dummy_img, torch.IntTensor([0]), 1


def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    return torch.stack(images, 0), torch.cat(targets), torch.IntTensor(target_lengths)


# --- 3. 模型定义 ---
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(256 * 2, NUM_CLASSES)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.cnn(x)
        x = self.avg_pool(x)
        b, c, h, w = x.size()
        x = x.view(b, c, w)
        x = x.permute(2, 0, 1)
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output


# --- 4. CTC 解码 ---
def ctc_decode(preds, blank=0):
    if preds.dim() == 3:
        preds = preds.permute(1, 0, 2)
    preds = preds.argmax(2)
    preds = preds.cpu().numpy()

    results = []
    for pred in preds:
        text = []
        prev = blank
        for p in pred:
            if p != blank and p != prev:
                text.append(idx2char.get(p, ''))
            prev = p
        results.append(''.join(text))
    return results


# --- 5. 编辑距离计算 ---
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# --- 6. 训练主程序 ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = IronDataset(TRAIN_LABEL_PATH, TRAIN_IMAGE_DIR, transform=transform)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = CRNN().to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_loss = float('inf')

    print(f"训练开始！使用设备: {device}")
    print(f"训练集: {train_size} 条 | 验证集: {val_size} 条")

    for epoch in range(100):
        model.train()
        total_loss = 0
        for imgs, targets, target_lens in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)

            preds = model(imgs)
            input_lens = torch.IntTensor([preds.size(0)] * imgs.size(0)).to(device)

            loss = criterion(preds.log_softmax(2), targets, input_lens, target_lens)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets, target_lens in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                target_lens = target_lens.to(device)

                preds = model(imgs)
                input_lens = torch.IntTensor([preds.size(0)] * imgs.size(0)).to(device)

                loss = criterion(preds.log_softmax(2), targets, input_lens, target_lens)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        scheduler.step()

        print(f"Epoch [{epoch + 1}/100]  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 保存最佳模型 (val_loss: {best_val_loss:.4f})")

    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")


# --- 7. 评估程序 ---
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model = CRNN().to(device)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"错误: 找不到模型文件 {MODEL_SAVE_PATH}，请先运行 train() 训练模型！")
        return

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    if not os.path.exists(TEST_CSV_PATH):
        print(f"错误: 找不到测试标签文件 {TEST_CSV_PATH}")
        return

    results = []
    correct_count = 0
    total_edit_distance = 0
    total_gt_chars = 0

    print(f"\n开始评估，测试文件: {TEST_CSV_PATH}")
    print("-" * 80)

    with open(TEST_CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue

            img_name = row[0].strip()
            gt_label = row[1].strip()
            img_path = os.path.join(TEST_IMAGE_DIR, img_name)

            if not os.path.exists(img_path):
                print(f"跳过: 找不到图片 {img_path}")
                continue

            img = Image.open(img_path).convert('L')
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(img)
                pred_text = ctc_decode(preds)[0]

            is_match = (pred_text == gt_label)
            if is_match:
                correct_count += 1
            else:
                print(f"错  {img_name:<<35s} | 预测: [{pred_text:<<25s}] | 答案: [{gt_label}]")

            edit_dist = levenshtein_distance(pred_text, gt_label)
            total_edit_distance += edit_dist
            total_gt_chars += len(gt_label)

            results.append({
                'image': img_name,
                'gt': gt_label,
                'pred': pred_text,
                'match': is_match,
                'edit_dist': edit_dist
            })

    total_samples = len(results)
    if total_samples == 0:
        print("没有成功评估任何样本，请检查路径和文件内容。")
        return

    sample_acc = correct_count / total_samples * 100
    char_acc = (total_gt_chars - total_edit_distance) / total_gt_chars * 100 if total_gt_chars > 0 else 0

    print("-" * 80)
    print(f"评估结果汇总:")
    print(f"  总样本数: {total_samples}")
    print(f"  完全正确: {correct_count}")
    print(f"  样本准确率 (Exact Match) : {sample_acc:.2f}%")
    print(f"  字符准确率 (Char Accuracy): {char_acc:.2f}%")
    print(f"  总编辑距离: {total_edit_distance}")
    print("-" * 80)

    return results


# --- 8. 单张/批量预测程序 ---
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model = CRNN().to(device)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"错误: 找不到模型文件 {MODEL_SAVE_PATH}，请先运行 train() 训练模型！")
        return

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    image_paths = (
        glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpg')) +
        glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpeg')) +
        glob.glob(os.path.join(TEST_IMAGE_DIR, '*.png')) +
        glob.glob(os.path.join(TEST_IMAGE_DIR, '*.bmp'))
    )

    if len(image_paths) == 0:
        print(f"警告: 在 {TEST_IMAGE_DIR} 中未找到任何图片")
        return

    print(f"\n找到 {len(image_paths)} 张待识别图片，开始识别...\n")

    results = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert('L')
            img = transform(img).unsqueeze(0).to(device)
            preds = model(img)
            text = ctc_decode(preds)[0]
            results.append((os.path.basename(img_path), text))
            print(f"{os.path.basename(img_path):<<20s} ->  {text}")

    return results


if __name__ == "__main__":
    # 首次运行请先训练模型
    # train()

    # 训练完成后，运行 evaluate() 与标准答案对比，获得准确率
    #evaluate()

    # 运行 predict() 仅输出识别结果
     predict()