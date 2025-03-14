// Main JavaScript for dog breed classifier web interface
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const predictionsContainer = document.getElementById('predictions');
    const loadingSpinner = document.querySelector('.loading');
    const errorMessage = document.querySelector('.error-message');

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropZone.classList.add('dragover');
    }

    function unhighlight() {
        dropZone.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                displayPreview(file);
                uploadAndPredict(file);
            } else {
                showError('Please upload an image file.');
            }
        }
    }

    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    async function uploadAndPredict(file) {
        const formData = new FormData();
        formData.append('file', file);

        showLoading();
        clearPredictions();
        hideError();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const result = await response.json();
            displayPredictions(result.predictions);
        } catch (error) {
            showError('Failed to get predictions. Please try again.');
        } finally {
            hideLoading();
        }
    }

    function displayPredictions(predictions) {
        predictionsContainer.innerHTML = '';
        
        predictions.forEach(pred => {
            const predItem = document.createElement('div');
            predItem.className = 'prediction-item';
            
            const confidence = (pred.confidence * 100).toFixed(2);
            const barWidth = Math.max(confidence, 1); // Ensure bar is visible even for tiny confidences
            
            predItem.innerHTML = `
                <div class="breed-name">${pred.breed} (${confidence}%)</div>
                <div class="confidence-bar">
                    <div class="confidence-level" style="width: ${barWidth}%"></div>
                </div>
            `;
            
            predictionsContainer.appendChild(predItem);
        });
    }

    function showLoading() {
        loadingSpinner.style.display = 'block';
    }

    function hideLoading() {
        loadingSpinner.style.display = 'none';
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    function hideError() {
        errorMessage.style.display = 'none';
    }

    function clearPredictions() {
        predictionsContainer.innerHTML = '';
    }
});
