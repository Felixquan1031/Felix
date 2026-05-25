#!/usr/bin/env python3
"""
初始化脚本：将 friend/ 目录下的现有照片导入 MongoDB。
运行前请确保已设置环境变量 MONGODB_URI。
"""

import os
import base64
import sys
from datetime import datetime
from pathlib import Path

# 尝试导入 pymongo
try:
    from pymongo import MongoClient
except ImportError:
    print("❌ 请先安装依赖: pip install pymongo[srv]")
    sys.exit(1)

MONGODB_URI = os.environ.get('MONGODB_URI')
if not MONGODB_URI:
    print("❌ 请设置环境变量 MONGODB_URI")
    print("   示例: export MONGODB_URI='mongodb+srv://用户名:密码@cluster0.xxxxx.mongodb.net/photo_album?retryWrites=true&w=majority'")
    sys.exit(1)

# 照片目录（相对于本脚本）
PHOTO_DIR = Path(__file__).parent.parent / 'friend'
if not PHOTO_DIR.exists():
    print(f"❌ 照片目录不存在: {PHOTO_DIR}")
    sys.exit(1)

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}


def main():
    print("🔗 连接 MongoDB...")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
    db = client['photo_album']
    photos_col = db['photos']

    # 清空现有数据（可选，如果你要重新导入）
    # photos_col.delete_many({})

    # 获取目录下所有图片
    image_files = [
        f for f in PHOTO_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ]

    if not image_files:
        print(f"❌ 在 {PHOTO_DIR} 中没有找到图片文件")
        sys.exit(1)

    print(f"📷 找到 {len(image_files)} 张照片，开始导入...\n")

    imported = 0
    skipped = 0

    for img_path in sorted(image_files):
        # 检查是否已存在（以文件名判断）
        title = img_path.stem
        existing = photos_col.find_one({'title': title})
        if existing:
            print(f"⏭️  已存在，跳过: {img_path.name}")
            skipped += 1
            continue

        # 读取并转 base64
        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        ext = img_path.suffix.lower().replace('.', '')
        if ext == 'jpg':
            ext = 'jpeg'
        mime = f"image/{ext}"
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:{mime};base64,{b64}"

        # 解析文件名中的日期（QQ图片格式）
        date_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        import re
        match = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})', img_path.name)
        if match:
            y, mo, d, h, mi = match.groups()
            date_str = f"{y}-{mo}-{d} {h}:{mi}"

        photo_doc = {
            'id': str(int(datetime.now().timestamp() * 1000) + imported),
            'title': title,
            'date': date_str,
            'imageBase64': data_url,
            'comments': []
        }

        photos_col.insert_one(photo_doc)
        size_kb = len(image_bytes) / 1024
        print(f"✅ 导入成功: {img_path.name} ({size_kb:.1f} KB)")
        imported += 1

    total = photos_col.count_documents({})
    print(f"\n🎉 完成！本次导入 {imported} 张，跳过 {skipped} 张。")
    print(f"📊 数据库中共有 {total} 张照片。")


if __name__ == '__main__':
    main()
