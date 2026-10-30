// State Management
const state = {
  activeTab: 'scanner',
  stream: null,
  isScanning: false,
  scanTimer: null,
  scanInterval: 300, // ms
  threshold: 0.363,
  soundEnabled: true,
  autoLog: true,
  lastSoundTime: 0,
  lastRecognizedUser: null,
  users: [],
  logs: [],
  enrollSnapBase64: null,
  addSampleTargetUserId: null
};

// DOM Elements
const videoEl = document.getElementById('video-element');
const overlayCanvas = document.getElementById('overlay-canvas');
const ctx = overlayCanvas.getContext('2d');
const videoPlaceholder = document.getElementById('video-placeholder');
const btnToggleCamera = document.getElementById('btn-toggle-camera');
const btnSingleSnap = document.getElementById('btn-single-snap');
const hudStatus = document.getElementById('hud-status');
const hudFps = document.getElementById('hud-fps');
const hudFacesCount = document.getElementById('hud-faces-count');
const hudList = document.getElementById('hud-detected-list');
const enrolledCountBadge = document.getElementById('enrolled-count-badge');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');
const soundToggle = document.getElementById('sound-toggle');
const autoLogToggle = document.getElementById('autolog-toggle');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSettings();
  loadSystemStatus();
  loadUsers();
  loadLogs();
  initEnrollment();
  initImageAnalyzer();
});

// Sound Effects via Web Audio API (Zero external files needed)
function playChime(isSuccess = true) {
  if (!state.soundEnabled) return;
  const now = Date.now();
  if (now - state.lastSoundTime < 2000) return; // throttle sounds to once per 2s
  state.lastSoundTime = now;

  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    const audioCtx = new AudioContext();

    if (isSuccess) {
      // Pleasant upbeat chime
      const osc1 = audioCtx.createOscillator();
      const osc2 = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc1.type = 'sine';
      osc2.type = 'triangle';
      osc1.frequency.setValueAtTime(523.25, audioCtx.currentTime); // C5
      osc1.frequency.setValueAtTime(659.25, audioCtx.currentTime + 0.1); // E5
      osc2.frequency.setValueAtTime(783.99, audioCtx.currentTime + 0.1); // G5

      gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(audioCtx.destination);

      osc1.start();
      osc2.start();
      osc1.stop(audioCtx.currentTime + 0.35);
      osc2.stop(audioCtx.currentTime + 0.35);
    } else {
      // Soft gentle low ping
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(329.63, audioCtx.currentTime); // E4
      osc.frequency.setValueAtTime(293.66, audioCtx.currentTime + 0.1); // D4

      gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start();
      osc.stop(audioCtx.currentTime + 0.3);
    }
  } catch (e) {
    console.debug('WebAudio not allowed yet:', e);
  }
}

// Navigation Tabs
function initNavigation() {
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  state.activeTab = tabId;
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  const activeBtn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
  const activeContent = document.getElementById(`tab-${tabId}`);
  if (activeBtn) activeBtn.classList.add('active');
  if (activeContent) activeContent.classList.add('active');

  // Pause or manage live scanner if leaving scanner tab
  if (tabId !== 'scanner' && state.isScanning) {
    stopLiveScan();
  }

  if (tabId === 'people') {
    loadUsers();
  } else if (tabId === 'logs') {
    loadLogs();
  }
}

// Toast Notifications
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const icons = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ'
  };

  toast.innerHTML = `
    <span style="font-weight: bold; font-size: 1.1rem;">${icons[type] || 'ℹ'}</span>
    <span>${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// Camera & Live Scanner
async function startCamera() {
  try {
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      }
    };
    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = state.stream;
    videoPlaceholder.style.display = 'none';

    await new Promise(resolve => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        syncCanvasDimensions();
        resolve();
      };
    });

    btnToggleCamera.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
      Stop Camera
    `;
    btnToggleCamera.classList.remove('btn-primary');
    btnToggleCamera.classList.add('btn-danger');

    startLiveScan();
    showToast('Camera started successfully', 'success');
  } catch (err) {
    console.error('Camera access error:', err);
    showToast('Failed to access camera: ' + err.message, 'error');
  }
}

function stopCamera() {
  stopLiveScan();
  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }
  videoEl.srcObject = null;
  videoPlaceholder.style.display = 'flex';
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  btnToggleCamera.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M4 4h3l2-2h6l2 2h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/><circle cx="12" cy="13" r="4"/></svg>
    Start Camera
  `;
  btnToggleCamera.classList.remove('btn-danger');
  btnToggleCamera.classList.add('btn-primary');

  hudStatus.textContent = 'Camera Off';
  hudFps.textContent = '-';
  hudFacesCount.textContent = '0';
  hudList.innerHTML = `<p style="font-size:0.85rem; color:var(--text-muted); text-align:center; padding:1rem;">Start camera to view live recognition</p>`;
}

function syncCanvasDimensions() {
  overlayCanvas.width = videoEl.videoWidth || 640;
  overlayCanvas.height = videoEl.videoHeight || 480;
}

btnToggleCamera.addEventListener('click', () => {
  if (state.stream) {
    stopCamera();
  } else {
    startCamera();
  }
});

btnSingleSnap.addEventListener('click', () => {
  if (!state.stream) {
    showToast('Please start the camera first', 'warning');
    return;
  }
  processLiveFrame(true);
});

function startLiveScan() {
  if (state.isScanning) return;
  state.isScanning = true;
  hudStatus.textContent = 'Scanning...';

  const loop = async () => {
    if (!state.isScanning || !state.stream) return;
    await processLiveFrame(false);
    state.scanTimer = setTimeout(loop, state.scanInterval);
  };
  loop();
}

function stopLiveScan() {
  state.isScanning = false;
  if (state.scanTimer) {
    clearTimeout(state.scanTimer);
    state.scanTimer = null;
  }
}

// Frame capture helper
function captureVideoFrameBase64() {
  if (!videoEl.videoWidth) return null;
  const offscreen = document.createElement('canvas');
  offscreen.width = videoEl.videoWidth;
  offscreen.height = videoEl.videoHeight;
  const offCtx = offscreen.getContext('2d');
  offCtx.drawImage(videoEl, 0, 0, offscreen.width, offscreen.height);
  return offscreen.toDataURL('image/jpeg', 0.85);
}

// Live frame processor
let isProcessingFrame = false;
async function processLiveFrame(manualSnap = false) {
  if (isProcessingFrame) return;
  const frameBase64 = captureVideoFrameBase64();
  if (!frameBase64) return;

  isProcessingFrame = true;
  const tStart = performance.now();

  try {
    const res = await fetch('/api/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: frameBase64,
        threshold: state.threshold,
        save_log: state.autoLog && manualSnap
      })
    });

    const data = await res.json();
    const elapsed = Math.round(performance.now() - tStart);

    hudFps.textContent = `${elapsed} ms`;
    hudFacesCount.textContent = data.faces_detected || 0;

    renderOverlayFaces(data.faces, data.image_size);
    updateHudDetectedList(data.faces);

    // Audio cue
    if (data.faces && data.faces.length > 0) {
      const anyMatch = data.faces.some(f => f.is_match);
      playChime(anyMatch);
    }
  } catch (err) {
    console.error('Recognition error:', err);
  } finally {
    isProcessingFrame = false;
  }
}

// Render bounding boxes & labels on the live camera canvas
function renderOverlayFaces(faces, imgSize) {
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!faces || faces.length === 0) return;

  const canvasW = overlayCanvas.width;
  const canvasH = overlayCanvas.height;
  const imgW = imgSize ? imgSize.width : canvasW;
  const imgH = imgSize ? imgSize.height : canvasH;

  const scaleX = canvasW / imgW;
  const scaleY = canvasH / imgH;

  faces.forEach(face => {
    const [origX, origY, origW, origH] = face.box;

    // Because webcam video is mirrored via CSS (scaleX(-1)),
    // calculate mirrored X coordinate on canvas:
    const mirroredX = imgW - (origX + origW);

    const x = mirroredX * scaleX;
    const y = origY * scaleY;
    const w = origW * scaleX;
    const h = origH * scaleY;

    const isMatch = face.is_match;
    const color = isMatch ? '#10b981' : '#ef4444'; // green / red
    const bgBadge = isMatch ? 'rgba(16, 185, 129, 0.9)' : 'rgba(239, 68, 68, 0.9)';

    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    const radius = 8;
    ctx.roundRect ? ctx.roundRect(x, y, w, h, radius) : ctx.rect(x, y, w, h);
    ctx.stroke();

    // Corner glow accents
    ctx.fillStyle = color;
    const cornerSize = Math.min(16, w * 0.2);
    // top-left
    ctx.fillRect(x - 1, y - 1, cornerSize, 4);
    ctx.fillRect(x - 1, y - 1, 4, cornerSize);
    // top-right
    ctx.fillRect(x + w - cornerSize + 1, y - 1, cornerSize, 4);
    ctx.fillRect(x + w - 3, y - 1, 4, cornerSize);
    // bottom-left
    ctx.fillRect(x - 1, y + h - 3, cornerSize, 4);
    ctx.fillRect(x - 1, y + h - cornerSize + 1, 4, cornerSize);
    // bottom-right
    ctx.fillRect(x + w - cornerSize + 1, y + h - 3, cornerSize, 4);
    ctx.fillRect(x + w - 3, y + h - cornerSize + 1, 4, cornerSize);

    // Draw facial landmark dots
    if (face.landmarks) {
      ctx.fillStyle = '#38bdf8';
      face.landmarks.forEach(pt => {
        const lx = (imgW - pt.x) * scaleX;
        const ly = pt.y * scaleY;
        ctx.beginPath();
        ctx.arc(lx, ly, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Name & Confidence Badge
    const label = isMatch
      ? `${face.name} • ${face.confidence}%`
      : `Unknown • ${face.confidence}%`;

    ctx.font = 'bold 14px -apple-system, BlinkMacSystemFont, sans-serif';
    const textMetrics = ctx.measureText(label);
    const badgeW = textMetrics.width + 20;
    const badgeH = 26;
    const badgeY = Math.max(0, y - badgeH - 6);

    ctx.fillStyle = bgBadge;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(x, badgeY, badgeW, badgeH, 6) : ctx.rect(x, badgeY, badgeW, badgeH);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x + 10, badgeY + 18);
  });
}

// Update HUD Detected Faces list
function updateHudDetectedList(faces) {
  if (!faces || faces.length === 0) {
    hudList.innerHTML = `<p style="font-size:0.85rem; color:var(--text-muted); text-align:center; padding:0.75rem;">No faces detected in view</p>`;
    return;
  }

  hudList.innerHTML = faces.map(face => {
    const isMatch = face.is_match;
    const cardClass = isMatch ? 'match' : 'unknown';
    const statusText = isMatch ? face.name : 'Unknown Person';
    const subText = isMatch
      ? `${face.confidence}% confidence (similarity: ${face.similarity})`
      : `No match found (best sim: ${face.similarity})`;

    return `
      <div class="detected-face-card ${cardClass}">
        <div style="font-size:1.4rem;">${isMatch ? '👤' : '❓'}</div>
        <div style="flex:1; overflow:hidden;">
          <div style="font-weight:700; font-size:0.95rem; text-overflow:ellipsis; white-space:nowrap; overflow:hidden;">${statusText}</div>
          <div style="font-size:0.75rem; color:var(--text-muted);">${subText}</div>
        </div>
      </div>
    `;
  }).join('');
}

// Enrollment Workflow
function initEnrollment() {
  const methodWebcam = document.getElementById('enroll-method-webcam');
  const methodFile = document.getElementById('enroll-method-file');
  const enrollWebcamArea = document.getElementById('enroll-webcam-area');
  const enrollFileArea = document.getElementById('enroll-file-area');
  const enrollVideo = document.getElementById('enroll-video');
  const btnStartEnrollCam = document.getElementById('btn-start-enroll-cam');
  const btnSnapEnrollCam = document.getElementById('btn-snap-enroll-cam');
  const enrollPreviewBox = document.getElementById('enroll-preview-box');
  const enrollPreviewImg = document.getElementById('enroll-preview-img');
  const enrollFileInput = document.getElementById('enroll-file-input');
  const enrollDropzone = document.getElementById('enroll-dropzone');
  const enrollForm = document.getElementById('enroll-form');
  const btnEnrollSubmit = document.getElementById('btn-enroll-submit');

  let enrollCamStream = null;

  // Toggle method radio
  methodWebcam.addEventListener('change', () => {
    enrollWebcamArea.style.display = 'block';
    enrollFileArea.style.display = 'none';
  });

  methodFile.addEventListener('change', () => {
    enrollWebcamArea.style.display = 'none';
    enrollFileArea.style.display = 'block';
    if (enrollCamStream) {
      enrollCamStream.getTracks().forEach(t => t.stop());
      enrollCamStream = null;
      btnStartEnrollCam.textContent = 'Start Camera';
    }
  });

  // Start Camera for enrollment
  btnStartEnrollCam.addEventListener('click', async () => {
    if (enrollCamStream) {
      enrollCamStream.getTracks().forEach(t => t.stop());
      enrollCamStream = null;
      enrollVideo.srcObject = null;
      btnStartEnrollCam.textContent = 'Start Camera';
      btnSnapEnrollCam.disabled = true;
      return;
    }

    try {
      enrollCamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      enrollVideo.srcObject = enrollCamStream;
      btnStartEnrollCam.textContent = 'Stop Camera';
      btnSnapEnrollCam.disabled = false;
    } catch (err) {
      showToast('Could not access camera: ' + err.message, 'error');
    }
  });

  // Snap photo from enrollment camera
  btnSnapEnrollCam.addEventListener('click', () => {
    if (!enrollVideo.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = enrollVideo.videoWidth;
    canvas.height = enrollVideo.videoHeight;
    const c = canvas.getContext('2d');
    c.drawImage(enrollVideo, 0, 0);

    state.enrollSnapBase64 = canvas.toDataURL('image/jpeg', 0.9);
    enrollPreviewImg.src = state.enrollSnapBase64;
    enrollPreviewBox.style.display = 'block';
    showToast('Photo captured! Ready to enroll.', 'success');
  });

  // File dropzone
  enrollDropzone.addEventListener('click', () => enrollFileInput.click());
  enrollDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    enrollDropzone.classList.add('dragover');
  });
  enrollDropzone.addEventListener('dragleave', () => enrollDropzone.classList.remove('dragover'));
  enrollDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    enrollDropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleEnrollFile(e.dataTransfer.files[0]);
    }
  });
  enrollFileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleEnrollFile(e.target.files[0]);
    }
  });

  function handleEnrollFile(file) {
    const reader = new FileReader();
    reader.onload = (evt) => {
      state.enrollSnapBase64 = evt.target.result;
      enrollPreviewImg.src = state.enrollSnapBase64;
      enrollPreviewBox.style.display = 'block';
      showToast(`Selected: ${file.name}`, 'info');
    };
    reader.readAsDataURL(file);
  }

  // Form submit
  enrollForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('enroll-name').value.trim();
    const role = document.getElementById('enroll-role').value.trim();

    if (!name) {
      showToast('Please enter person name', 'warning');
      return;
    }
    if (!state.enrollSnapBase64) {
      showToast('Please capture or select a photo first', 'warning');
      return;
    }

    btnEnrollSubmit.disabled = true;
    btnEnrollSubmit.innerHTML = `<span>Enrolling face...</span>`;

    try {
      const res = await fetch('/api/users/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          role: role || 'Member',
          image: state.enrollSnapBase64
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Enrollment failed');
      }

      showToast(`Success! ${name} enrolled.`, 'success');
      playChime(true);

      // Reset form
      enrollForm.reset();
      state.enrollSnapBase64 = null;
      enrollPreviewBox.style.display = 'none';

      if (enrollCamStream) {
        enrollCamStream.getTracks().forEach(t => t.stop());
        enrollCamStream = null;
        enrollVideo.srcObject = null;
        btnStartEnrollCam.textContent = 'Start Camera';
        btnSnapEnrollCam.disabled = true;
      }

      // Refresh directory and switch to people tab
      await loadUsers();
      loadSystemStatus();
      switchTab('people');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      btnEnrollSubmit.disabled = false;
      btnEnrollSubmit.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/></svg>
        Complete Enrollment
      `;
    }
  });
}

// People Directory Management
async function loadUsers() {
  const grid = document.getElementById('people-grid');
  try {
    const res = await fetch('/api/users');
    const data = await res.json();
    state.users = data.users || [];

    enrolledCountBadge.textContent = state.users.length;

    if (state.users.length === 0) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-state-icon">👥</div>
          <h3>No Enrolled People Yet</h3>
          <p>Add people using the "Enroll Person" tab to enable face recognition.</p>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="switchTab('enroll')">Enroll First Person</button>
        </div>
      `;
      return;
    }

    grid.innerHTML = state.users.map(user => {
      const thumb = user.thumbnail_path || 'https://via.placeholder.com/90?text=Face';
      const createdDate = new Date(user.created_at).toLocaleDateString();

      return `
        <div class="person-card">
          <img src="${thumb}" class="person-avatar" alt="${user.name}" onerror="this.src='https://via.placeholder.com/90?text=Face'" />
          <div class="person-name">${user.name}</div>
          <div class="person-role">${user.role || 'Member'}</div>
          <div class="person-meta">
            <span>${user.sample_count || 1} sample(s)</span> • <span>${createdDate}</span>
          </div>
          <div class="person-actions">
            <button class="btn btn-secondary btn-sm" onclick="promptAddSample(${user.id}, '${user.name.replace(/'/g, "\\'")}')">
              + Sample
            </button>
            <button class="btn btn-danger btn-sm" onclick="deleteUser(${user.id}, '${user.name.replace(/'/g, "\\'")}')">
              Delete
            </button>
          </div>
        </div>
      `;
    }).join('');
  } catch (err) {
    console.error('Error loading users:', err);
    showToast('Failed to load user directory', 'error');
  }
}

async function deleteUser(userId, userName) {
  if (!confirm(`Are you sure you want to delete ${userName}? All face embeddings will be removed.`)) {
    return;
  }

  try {
    const res = await fetch(`/api/users/${userId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Deletion failed');
    showToast(`Deleted ${userName}`, 'info');
    loadUsers();
    loadSystemStatus();
  } catch (err) {
    showToast('Error deleting user: ' + err.message, 'error');
  }
}

// Add sample prompt
function promptAddSample(userId, userName) {
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*';
  fileInput.onchange = async (e) => {
    if (!e.target.files.length) return;
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = async (evt) => {
      try {
        showToast(`Adding sample for ${userName}...`, 'info');
        const res = await fetch('/api/users/add-sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            image: evt.target.result
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to add sample');
        showToast(`Added new sample for ${userName}!`, 'success');
        loadUsers();
      } catch (err) {
        showToast(err.message, 'error');
      }
    };
    reader.readAsDataURL(file);
  };
  fileInput.click();
}

// Static Image Analyzer Tab
function initImageAnalyzer() {
  const dropzone = document.getElementById('analyze-dropzone');
  const fileInput = document.getElementById('analyze-file-input');
  const resultCard = document.getElementById('analyze-result-card');
  const canvas = document.getElementById('analyze-canvas');
  const actx = canvas.getContext('2d');
  const detailsDiv = document.getElementById('analyze-details');

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) analyzeFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) analyzeFile(e.target.files[0]);
  });

  async function analyzeFile(file) {
    showToast('Analyzing image...', 'info');
    const reader = new FileReader();
    reader.onload = async (evt) => {
      const base64 = evt.target.result;
      const img = new Image();
      img.onload = async () => {
        // Draw image on canvas
        canvas.width = img.width;
        canvas.height = img.height;
        actx.drawImage(img, 0, 0);

        try {
          const res = await fetch('/api/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image: base64,
              threshold: state.threshold,
              save_log: true
            })
          });

          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || 'Analysis error');

          resultCard.style.display = 'block';

          // Draw boxes
          if (data.faces && data.faces.length > 0) {
            data.faces.forEach(face => {
              const [x, y, w, h] = face.box;
              const isMatch = face.is_match;
              const color = isMatch ? '#10b981' : '#ef4444';

              actx.strokeStyle = color;
              actx.lineWidth = Math.max(3, Math.round(canvas.width / 300));
              actx.strokeRect(x, y, w, h);

              // Label
              const label = isMatch
                ? `${face.name} (${face.confidence}%)`
                : `Unknown (${face.confidence}%)`;

              actx.font = `bold ${Math.max(14, Math.round(canvas.width / 40))}px sans-serif`;
              actx.fillStyle = color;
              actx.fillText(label, x + 6, Math.max(25, y - 8));
            });

            detailsDiv.innerHTML = `
              <h4 style="margin-bottom:0.75rem;">Results (${data.faces_detected} face(s) found in ${data.processing_time_ms} ms):</h4>
              <div style="display:flex; flex-direction:column; gap:0.5rem;">
                ${data.faces.map(f => `
                  <div style="padding:0.6rem 0.85rem; border-radius:8px; background:rgba(255,255,255,0.05); border-left:4px solid ${f.is_match ? '#10b981' : '#ef4444'};">
                    <strong>${f.name}</strong> • ${f.confidence}% confidence
                    <div style="font-size:0.8rem; color:var(--text-muted);">Cosine Similarity: ${f.similarity} (Threshold: ${state.threshold})</div>
                  </div>
                `).join('')}
              </div>
            `;
          } else {
            detailsDiv.innerHTML = `<p style="color:var(--text-muted);">No faces detected in this image.</p>`;
          }

          showToast(`Analysis complete: ${data.faces_detected} face(s) detected.`, 'success');
        } catch (err) {
          showToast(err.message, 'error');
        }
      };
      img.src = base64;
    };
    reader.readAsDataURL(file);
  }
}

// Logs and Attendance
async function loadLogs() {
  const tbody = document.getElementById('logs-table-body');
  try {
    const res = await fetch('/api/logs?limit=100');
    const data = await res.json();
    state.logs = data.logs || [];

    renderLogsTable(state.logs);
  } catch (err) {
    console.error('Error loading logs:', err);
  }
}

function renderLogsTable(logs) {
  const tbody = document.getElementById('logs-table-body');
  if (!logs || logs.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align:center; padding:2rem; color:var(--text-muted);">
          No recognition activity logged yet.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = logs.map(item => {
    const snap = item.snapshot_path || 'https://via.placeholder.com/36?text=Face';
    const isKnown = item.user_id !== null && item.user_name !== 'Unknown';
    const badgeColor = isKnown ? 'var(--success)' : 'var(--danger)';
    const dateFormatted = new Date(item.created_at).toLocaleString();

    return `
      <tr>
        <td>
          <img src="${snap}" class="table-avatar" alt="Face" onerror="this.src='https://via.placeholder.com/36?text=Face'" />
        </td>
        <td style="font-weight:600;">${item.user_name}</td>
        <td>
          <span style="display:inline-block; padding:0.2rem 0.5rem; border-radius:999px; font-size:0.75rem; font-weight:700; color:#fff; background:${badgeColor};">
            ${isKnown ? 'Recognized' : 'Unknown'}
          </span>
        </td>
        <td>${item.confidence}%</td>
        <td><code>${item.similarity_score}</code></td>
        <td style="color:var(--text-muted); font-size:0.85rem;">${dateFormatted}</td>
      </tr>
    `;
  }).join('');
}

// Logs filtering
document.getElementById('log-search-input').addEventListener('input', (e) => {
  const q = e.target.value.toLowerCase();
  const filtered = state.logs.filter(l => l.user_name.toLowerCase().includes(q));
  renderLogsTable(filtered);
});

// Clear logs
document.getElementById('btn-clear-logs').addEventListener('click', async () => {
  if (!confirm('Are you sure you want to clear all recognition history?')) return;
  try {
    await fetch('/api/logs', { method: 'DELETE' });
    showToast('Recognition logs cleared', 'info');
    loadLogs();
  } catch (err) {
    showToast('Failed to clear logs: ' + err.message, 'error');
  }
});

// Settings & Thresholds
function initSettings() {
  thresholdSlider.addEventListener('input', (e) => {
    state.threshold = parseFloat(e.target.value);
    thresholdVal.textContent = state.threshold.toFixed(3);
  });

  soundToggle.addEventListener('change', (e) => {
    state.soundEnabled = e.target.checked;
    showToast(`Audio feedback ${state.soundEnabled ? 'enabled' : 'muted'}`, 'info');
  });

  autoLogToggle.addEventListener('change', (e) => {
    state.autoLog = e.target.checked;
  });

  // Dark/Light Mode Theme Toggle
  const themeToggle = document.getElementById('theme-toggle');
  themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    themeToggle.textContent = newTheme === 'light' ? '🌙' : '☀️';
  });
}

// System Status Badge
async function loadSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const badge = document.getElementById('system-status-badge');
    badge.innerHTML = `<span class="status-dot"></span> YuNet & SFace Ready (${data.enrolled_users_count} enrolled)`;
  } catch (e) {
    console.debug('Status load failed:', e);
  }
}
