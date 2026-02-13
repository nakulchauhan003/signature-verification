// File upload handling
let referenceFile = null;
let testFile = null;

const dropZone1 = document.getElementById('dropZone1');
const dropZone2 = document.getElementById('dropZone2');
const referenceInput = document.getElementById('referenceInput');
const testInput = document.getElementById('testInput');
const verifyBtn = document.getElementById('verifyBtn');
const loading = document.getElementById('loading');
const result = document.getElementById('result');

// Setup drag and drop for reference signature
setupDropZone(dropZone1, referenceInput, 'preview1', (file) => {
    referenceFile = file;
    checkBothFiles();
});

// Setup drag and drop for test signature
setupDropZone(dropZone2, testInput, 'preview2', (file) => {
    testFile = file;
    checkBothFiles();
});

function setupDropZone(dropZone, input, previewId, callback) {
    // Click to upload
    dropZone.addEventListener('click', () => input.click());

    // File input change
    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file, previewId, callback);
        }
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file, previewId, callback);
        }
    });
}

function handleFile(file, previewId, callback) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById(previewId);
        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
    };
    reader.readAsDataURL(file);
    callback(file);
}

function checkBothFiles() {
    verifyBtn.disabled = !(referenceFile && testFile);
}

// Verify button click
verifyBtn.addEventListener('click', async () => {
    if (!referenceFile || !testFile) return;

    // Show loading
    loading.style.display = 'block';
    result.style.display = 'none';
    verifyBtn.disabled = true;

    // Prepare form data
    const formData = new FormData();
    formData.append('reference', referenceFile);
    formData.append('test', testFile);

    try {
        const response = await fetch('/api/verify', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResult(data);
        } else {
            displayError(data.error || 'Verification failed');
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        loading.style.display = 'none';
        verifyBtn.disabled = false;
    }
});

function displayResult(data) {
    const isGenuine = data.is_genuine;

    result.className = 'result ' + (isGenuine ? 'genuine' : 'forged');
    result.innerHTML = `
        <h2>${data.verdict}</h2>
        <p style="font-size: 1.2rem; margin-bottom: 10px;">
            ${isGenuine ? '✅ Signatures Match!' : '❌ Signatures Do Not Match'}
        </p>
        <div class="result-details">
            <div class="result-item">
                <label>Similarity</label>
                <value>${data.similarity.toFixed(2)}%</value>
            </div>
            <div class="result-item">
                <label>Confidence</label>
                <value>${data.confidence.toFixed(2)}%</value>
            </div>
            <div class="result-item">
                <label>Threshold</label>
                <value>${data.threshold.toFixed(0)}%</value>
            </div>
        </div>
    `;
    result.style.display = 'block';
}

function displayError(message) {
    result.className = 'result forged';
    result.innerHTML = `
        <h2>Error</h2>
        <p>${message}</p>
    `;
    result.style.display = 'block';
}