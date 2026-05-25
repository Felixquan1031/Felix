// ===== 配置 =====
const API_BASE = window.location.origin.includes('localhost')
    ? 'http://localhost:3000/api'
    : '/api';

// ===== DOM 元素 =====
const gallery = document.getElementById('gallery');
const photoCount = document.getElementById('photoCount');
const loading = document.getElementById('loading');
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightboxImg');
const photoTitle = document.getElementById('photoTitle');
const photoDate = document.getElementById('photoDate');
const commentsList = document.getElementById('commentsList');
const commentCount = document.getElementById('commentCount');
const commentName = document.getElementById('commentName');
const commentText = document.getElementById('commentText');
const submitComment = document.getElementById('submitComment');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const refreshBtn = document.getElementById('refreshBtn');
const toast = document.getElementById('toast');

let photos = [];
let currentIndex = 0;

// ===== 工具函数 =====
function showToast(message, type = '') {
    toast.textContent = message;
    toast.className = 'toast ' + type;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== API 调用 =====
async function fetchPhotos() {
    try {
        loading.style.display = 'block';
        gallery.innerHTML = '';
        const res = await fetch(`${API_BASE}/photos`);
        if (!res.ok) throw new Error('加载失败');
        photos = await res.json();
        renderGallery();
    } catch (err) {
        showToast('加载照片失败: ' + err.message, 'error');
        console.error(err);
    } finally {
        loading.style.display = 'none';
    }
}

async function uploadPhoto(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async () => {
            try {
                const base64 = reader.result;
                // 限制大小检查（base64 约比原文件大 33%）
                if (base64.length > 2.8 * 1024 * 1024) {
                    throw new Error('图片太大，请压缩后重试');
                }
                const title = file.name.replace(/\.[^.]+$/, '');
                const res = await fetch(`${API_BASE}/photos`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title, imageBase64: base64 })
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.error || '上传失败');
                }
                resolve(await res.json());
            } catch (e) {
                reject(e);
            }
        };
        reader.onerror = () => reject(new Error('读取文件失败'));
        reader.readAsDataURL(file);
    });
}

async function postComment(photoId, name, text) {
    const res = await fetch(`${API_BASE}/comments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photoId, name, text })
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || '评论失败');
    }
    return res.json();
}

// ===== 渲染相册 =====
function renderGallery() {
    photoCount.textContent = photos.length;

    if (photos.length === 0) {
        gallery.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📷</div>
                <h3>还没有照片</h3>
                <p>点击上方上传区域，添加第一张照片吧！</p>
            </div>
        `;
        return;
    }

    gallery.innerHTML = photos.map((photo, index) => {
        const comments = photo.comments || [];
        return `
            <article class="photo-card" data-index="${index}">
                <img class="photo-thumb" src="${escapeHtml(photo.imageBase64)}"
                     alt="${escapeHtml(photo.title)}" loading="lazy">
                <div class="photo-info">
                    <div class="photo-name">${escapeHtml(photo.title)}</div>
                    <div class="photo-meta">
                        <span>${photo.date || '-'}</span>
                        ${comments.length > 0 ? `
                            <span class="comment-badge">💬 ${comments.length}</span>
                        ` : ''}
                    </div>
                </div>
            </article>
        `;
    }).join('');

    document.querySelectorAll('.photo-card').forEach(card => {
        card.addEventListener('click', () => openLightbox(parseInt(card.dataset.index)));
    });
}

// ===== 灯箱 =====
function openLightbox(index) {
    currentIndex = index;
    updateLightbox();
    lightbox.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    lightbox.classList.remove('active');
    document.body.style.overflow = '';
}

function updateLightbox() {
    const photo = photos[currentIndex];
    lightboxImg.src = photo.imageBase64;
    photoTitle.textContent = photo.title;
    photoDate.textContent = photo.date || '-';
    renderComments(photo);
}

function nextPhoto() {
    currentIndex = (currentIndex + 1) % photos.length;
    updateLightbox();
}

function prevPhoto() {
    currentIndex = (currentIndex - 1 + photos.length) % photos.length;
    updateLightbox();
}

// ===== 评论 =====
function renderComments(photo) {
    const comments = photo.comments || [];
    commentCount.textContent = comments.length;

    if (comments.length === 0) {
        commentsList.innerHTML = `
            <div class="empty-comments">
                <p>暂无评论，来做第一个评论的人吧！</p>
            </div>
        `;
        return;
    }

    commentsList.innerHTML = comments.map(c => `
        <div class="comment-item">
            <div class="comment-header">
                <span class="comment-author">${escapeHtml(c.name)}</span>
                <span class="comment-time">${escapeHtml(c.time)}</span>
            </div>
            <div class="comment-text">${escapeHtml(c.text)}</div>
        </div>
    `).join('');

    commentsList.scrollTop = commentsList.scrollHeight;
}

async function handleSubmitComment() {
    const name = commentName.value;
    const text = commentText.value;

    if (!text.trim()) {
        commentText.focus();
        return;
    }

    submitComment.disabled = true;
    submitComment.textContent = '发送中...';

    try {
        const photo = photos[currentIndex];
        await postComment(photo.id, name, text);

        // 更新本地数据并重新渲染
        if (!photo.comments) photo.comments = [];
        photo.comments.push({
            name: name.trim() || '匿名用户',
            text: text.trim(),
            time: new Date().toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
        });

        commentText.value = '';
        renderComments(photo);
        renderGallery();
        showToast('评论发表成功！', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        submitComment.disabled = false;
        submitComment.textContent = '发表评论';
    }
}

// ===== 上传处理 =====
async function handleFiles(files) {
    const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
    if (imageFiles.length === 0) {
        showToast('请选择图片文件', 'error');
        return;
    }

    uploadArea.style.display = 'none';
    uploadProgress.style.display = 'flex';

    for (let i = 0; i < imageFiles.length; i++) {
        const file = imageFiles[i];
        progressText.textContent = `正在上传 ${i + 1}/${imageFiles.length}: ${file.name}`;
        progressFill.style.width = `${(i / imageFiles.length) * 100}%`;

        try {
            await uploadPhoto(file);
            showToast(`「${file.name}」上传成功`, 'success');
        } catch (err) {
            showToast(`「${file.name}」上传失败: ${err.message}`, 'error');
        }
    }

    progressFill.style.width = '100%';
    progressText.textContent = '上传完成！';

    setTimeout(() => {
        uploadArea.style.display = 'block';
        uploadProgress.style.display = 'none';
        progressFill.style.width = '0%';
        fetchPhotos();
    }, 1500);
}

// ===== 事件监听 =====
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFiles(e.target.files));

uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

document.querySelector('.lightbox-close').addEventListener('click', closeLightbox);
lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });
document.querySelector('.lightbox-nav.next').addEventListener('click', e => { e.stopPropagation(); nextPhoto(); });
document.querySelector('.lightbox-nav.prev').addEventListener('click', e => { e.stopPropagation(); prevPhoto(); });

document.addEventListener('keydown', e => {
    if (!lightbox.classList.contains('active')) return;
    if (e.key === 'Escape') closeLightbox();
    if (e.key === 'ArrowRight') nextPhoto();
    if (e.key === 'ArrowLeft') prevPhoto();
});

submitComment.addEventListener('click', handleSubmitComment);
commentText.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmitComment();
    }
});

refreshBtn.addEventListener('click', fetchPhotos);

// ===== 初始化 =====
fetchPhotos();
