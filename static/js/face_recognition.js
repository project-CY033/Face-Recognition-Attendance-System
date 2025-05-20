/**
 * Face detection client-side helper
 * Note: This only handles the webcam capture and UI.
 * Actual face recognition is done on the server.
 */
class FaceRecognition {
    constructor(webcamCapture, options = {}) {
        this.webcam = webcamCapture;
        this.detectionInterval = options.detectionInterval || 500; // ms
        this.detectionThreshold = options.detectionThreshold || 3; // consecutive detections
        this.detectTimer = null;
        this.faceDetected = false;
        this.consecutiveDetections = 0;
        this.captureButton = options.captureButton || null;
        this.statusElement = options.statusElement || null;
        this.readyToCapture = false;
        
        // Bind event handlers
        if (this.captureButton) {
            this.captureButton.addEventListener('click', this.handleCapture.bind(this));
        }
        
        // Set callback handlers
        this.onFaceDetected = this.handleFaceDetected.bind(this);
        this.onNoFaceDetected = this.handleNoFaceDetected.bind(this);
        this.onMultipleFacesDetected = this.handleMultipleFacesDetected.bind(this);
        this.onCapture = options.onCapture || null;
        
        // Set webcam callbacks
        this.webcam.onFaceDetected = this.onFaceDetected;
        this.webcam.onNoFaceDetected = this.onNoFaceDetected;
        this.webcam.onMultipleFacesDetected = this.onMultipleFacesDetected;
    }
    
    /**
     * Start face detection
     */
    start() {
        this.detectTimer = setInterval(this.detectFace.bind(this), this.detectionInterval);
        this.updateStatus('Looking for face...');
    }
    
    /**
     * Stop face detection
     */
    stop() {
        if (this.detectTimer) {
            clearInterval(this.detectTimer);
            this.detectTimer = null;
        }
    }
    
    /**
     * Detect face in current frame
     */
    async detectFace() {
        try {
            // Get frame as blob
            const frameBlob = await this.webcam.getFrameBlob();
            
            // Send to server for face detection
            const formData = new FormData();
            formData.append('frame', frameBlob);
            
            const response = await fetch('/detect-face', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Face detection request failed');
            }
            
            const data = await response.json();
            
            if (data.faces && Array.isArray(data.faces)) {
                // Draw face boxes on canvas
                this.webcam.drawFaceBoxes(data.faces);
            }
        } catch (error) {
            console.error('Face detection error:', error);
        }
    }
    
    /**
     * Handle face detected event
     * @param {Array} face Face detection box [top, right, bottom, left]
     */
    handleFaceDetected(face) {
        this.consecutiveDetections++;
        
        if (this.consecutiveDetections >= this.detectionThreshold && !this.readyToCapture) {
            this.readyToCapture = true;
            this.updateStatus('Face detected! Click capture to continue');
            if (this.captureButton) {
                this.captureButton.disabled = false;
            }
        }
    }
    
    /**
     * Handle no face detected event
     */
    handleNoFaceDetected() {
        this.consecutiveDetections = 0;
        this.readyToCapture = false;
        this.updateStatus('No face detected. Please position your face in the frame');
        if (this.captureButton) {
            this.captureButton.disabled = true;
        }
    }
    
    /**
     * Handle multiple faces detected event
     */
    handleMultipleFacesDetected(faces) {
        this.consecutiveDetections = 0;
        this.readyToCapture = false;
        this.updateStatus('Multiple faces detected. Please ensure only your face is in the frame');
        if (this.captureButton) {
            this.captureButton.disabled = true;
        }
    }
    
    /**
     * Handle capture button click
     */
    handleCapture() {
        if (!this.readyToCapture) {
            this.updateStatus('Please position your face properly before capturing');
            return;
        }
        
        // Capture image
        const imageData = this.webcam.capture();
        
        // Stop detection
        this.stop();
        
        // Call callback if provided
        if (this.onCapture && typeof this.onCapture === 'function') {
            this.onCapture(imageData);
        }
    }
    
    /**
     * Update status message
     * @param {string} message Status message
     */
    updateStatus(message) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
        }
    }
}
