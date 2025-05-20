/**
 * Webcam capture utility for face recognition
 */
class WebcamCapture {
    constructor(videoElement, canvasElement, options = {}) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.streaming = false;
        this.facingMode = options.facingMode || 'user'; // 'user' for front camera, 'environment' for back
        this.stream = null;
        this.faceDetectionBoxes = [];

        // Optional callback functions
        this.onFaceDetected = options.onFaceDetected || null;
        this.onNoFaceDetected = options.onNoFaceDetected || null;
        this.onMultipleFacesDetected = options.onMultipleFacesDetected || null;
    }

    /**
     * Start webcam stream
     */
    async start() {
        try {
            // Set up camera options
            const constraints = {
                video: {
                    facingMode: this.facingMode,
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            };

            // Get access to webcam
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Set video source
            this.video.srcObject = this.stream;
            this.video.play();

            // Set canvas size once video is loaded
            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.streaming = true;
            });

            return true;
        } catch (error) {
            console.error('Error starting webcam:', error);
            return false;
        }
    }

    /**
     * Stop webcam stream
     */
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.streaming = false;
        }
    }

    /**
     * Switch camera between front and back (if available)
     */
    async switchCamera() {
        this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
        this.stop();
        await this.start();
    }

    /**
     * Capture current frame from webcam
     * @returns {string} Base64 encoded image data
     */
    capture() {
        if (!this.streaming) {
            console.error('Webcam not streaming');
            return null;
        }

        // Draw current video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert canvas to data URL
        return this.canvas.toDataURL('image/jpeg');
    }

    /**
     * Draw face detection boxes on canvas
     * @param {Array} faceBoxes Array of face detection boxes [top, right, bottom, left]
     */
    drawFaceBoxes(faceBoxes) {
        if (!this.streaming) return;

        // Clear previous drawings
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw current video frame
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Draw face boxes
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 3;
        
        faceBoxes.forEach(box => {
            const [top, right, bottom, left] = box;
            const width = right - left;
            const height = bottom - top;
            
            this.ctx.strokeRect(left, top, width, height);
        });

        // Store current face boxes
        this.faceDetectionBoxes = faceBoxes;

        // Call appropriate callback
        if (faceBoxes.length === 0 && this.onNoFaceDetected) {
            this.onNoFaceDetected();
        } else if (faceBoxes.length === 1 && this.onFaceDetected) {
            this.onFaceDetected(faceBoxes[0]);
        } else if (faceBoxes.length > 1 && this.onMultipleFacesDetected) {
            this.onMultipleFacesDetected(faceBoxes);
        }
    }

    /**
     * Get raw frame data for face detection processing
     * @returns {Blob} JPEG blob of current frame
     */
    async getFrameBlob() {
        return new Promise((resolve, reject) => {
            this.canvas.toBlob(blob => {
                if (blob) resolve(blob);
                else reject(new Error('Failed to convert canvas to blob'));
            }, 'image/jpeg');
        });
    }
}
