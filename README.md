# 📸 我们的相册 - 协作版

一个支持多人共享评论、上传照片的在线相册，部署在 Vercel + MongoDB Atlas（均免费）。

---

## 🚀 部署步骤

### 第一步：注册 MongoDB Atlas（数据库）

1. 打开 https://www.mongodb.com/atlas
2. 点击 **"Try Free"**，用邮箱/Google 账号注册
3. 创建一个新的 **Shared Cluster**（免费版，不用改任何配置，直接点 Create）
4. 等待集群创建完成（约 1-3 分钟）
5. 进入 **Database Access**，点击 **"Add New Database User"**：
   - 用户名：`photo_user`
   - 密码：自己设一个（记下来！）
   - 权限：**Read and write to any database**
   - 点击 **Add User**
6. 进入 **Network Access**，点击 **"Add IP Address"**：
   - 选择 **"Allow Access from Anywhere"**（输入 `0.0.0.0/0`）
   - 点击 **Confirm**
7. 回到 **Database** 页面，点击 **"Connect"** → **"Drivers"**
8. 选择 **Python**，复制连接字符串，格式如下：
   ```
   mongodb+srv://photo_user:你的密码@cluster0.xxxxx.mongodb.net/photo_album?retryWrites=true&w=majority
   ```
   **把 `<password>` 替换成你刚才设的密码，保存好这个字符串！**

---

### 第二步：导入现有照片到数据库

1. 确保你电脑上安装了 Python 和 pip
2. 安装依赖：
   ```bash
   pip install pymongo
   ```
3. 设置环境变量（PowerShell）：
   ```powershell
   $env:MONGODB_URI="mongodb+srv://photo_user:你的密码@cluster0.xxxxx.mongodb.net/photo_album?retryWrites=true&w=majority"
   ```
4. 运行初始化脚本：
   ```bash
   cd friend-collab
   python init_data.py
   ```
   看到 "🎉 完成！" 就表示照片已导入数据库。

---

### 第三步：注册并部署到 Vercel

1. 打开 https://vercel.com
2. 点击 **"Sign Up"**，用 **GitHub 账号** 一键登录（推荐）
3. 回到你电脑，把 `friend-collab` 文件夹推送到 GitHub：
   ```bash
   cd friend-collab
   git init
   git add .
   git commit -m "init photo album"
   ```
   然后到 GitHub 创建一个空仓库（不要勾选 README），按页面提示把代码 push 上去：
   ```bash
   git remote add origin https://github.com/你的用户名/仓库名.git
   git branch -M main
   git push -u origin main
   ```
4. 回到 Vercel，点击 **"Add New Project"**
5. 选择你刚才创建的 GitHub 仓库，点击 **Import**
6. 配置项目：
   - **Framework Preset**: 选 `Other`
   - **Root Directory**: 保持默认（`.`）
7. 点击 **"Environment Variables"**，添加：
   - **Name**: `MONGODB_URI`
   - **Value**: 你刚才保存的 MongoDB 连接字符串
8. 点击 **Deploy**
9. 等待 1-2 分钟，部署完成后 Vercel 会给你一个 `.vercel.app` 网址，这就是你们的相册地址！

---

## ✨ 功能

- 📤 **上传照片**：点击或拖拽上传，所有人都能看到
- 💬 **共享评论**：评论存在数据库里，所有朋友看的是同一份
- 🖼 **灯箱预览**：点击照片看大图，支持键盘左右切换
- 📱 **手机适配**：响应式设计，手机也能用

---

## 🔧 本地开发测试

如果你想在本地测试：

```bash
cd friend-collab
pip install pymongo
$env:MONGODB_URI="你的连接字符串"
python -m http.server 8080
```

然后浏览器打开 `http://localhost:8080`。

**注意**：本地前端调用 API 时需要解决跨域，建议直接部署到 Vercel 测试。

---

## ❓ 常见问题

**Q: 上传照片有大小限制吗？**
A: 建议单张不超过 2MB。免费 MongoDB 有 512MB 容量，大约能存 500-1000 张照片。

**Q: 朋友怎么上传照片？**
A: 把 Vercel 网址发给朋友，他们打开网页直接拖拽或点击上传即可，无需登录。

**Q: 数据安全吗？**
A: MongoDB Atlas 和 Vercel 都是正规云服务。连接字符串不要泄露给不信任的人。

**Q: 国内访问慢吗？**
A: Vercel 在国内部分地区访问可能较慢。如果太慢，可以考虑：
   - 在 Vercel 设置里开启 **Edge Network**
   - 或者后期迁移到国内云（需要备案）
