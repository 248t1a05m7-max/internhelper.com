async function startCamera(videoId) {
  const video = document.getElementById(videoId);
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
  } catch (err) {
    alert('Camera permission is required for face attendance.');
    console.error(err);
  }
}

function captureFace(videoId, canvasId, outputInputId = null) {
  const video = document.getElementById(videoId);
  const canvas = document.getElementById(canvasId);

  if (!video || !canvas || !video.videoWidth) {
    return null;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
  if (outputInputId) {
    const output = document.getElementById(outputInputId);
    if (output) output.value = dataUrl;
  }
  return dataUrl;
}
