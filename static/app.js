const form = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const statusBox = document.getElementById('status');
const resultOutput = document.getElementById('resultOutput');
const polygonCount = document.getElementById('polygonCount');
const metricParcels = document.getElementById('metricParcels');
const processedImage = document.getElementById('processedImage');

function updateStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.style.color = isError ? '#f87171' : '#94a3b8';
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];

  if (!file) {
    updateStatus('Please choose an image first.', true);
    return;
  }

  updateStatus('Processing image...');
  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Processing failed.');

    processedImage.src = `${data.image_url}?t=${Date.now()}`;
    const tally = Number.isFinite(data.vision_estimate) ? data.vision_estimate : data.count || 0;
    polygonCount.textContent = `Vision estimate: ${tally} parcel-like regions`;
    metricParcels.textContent = String(tally);
    resultOutput.textContent = JSON.stringify(data, null, 2);
    updateStatus(`Vision estimate complete: ${tally} parcel-like regions. AI model not installed.`);
  } catch (error) {
    updateStatus(error.message, true);
    resultOutput.textContent = JSON.stringify({ error: error.message }, null, 2);
  }
});
