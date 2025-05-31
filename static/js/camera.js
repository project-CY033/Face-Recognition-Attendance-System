/**
 * Enhanced Camera Utilities for SmartAttendance
 * Handles camera initialization, face detection, and photo capture
 */

class CameraManager {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.context = null;
        this.stream = null;
        this.isInitialized = false;
        this.faceDetectionEnabled = false;
        
        // Configuration
        this.config = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            captureFormat: 'image/jpeg',
            captureQuality: 0.9
        };
        
        // Bind methods
        this.initializeCamera = this.initializeCamera.bind(this);
        this.stopCamera = this.stopCamera.bind(this);
        this.capturePhoto = this.capturePhoto.bind(this);
    }

    /**
     * Initialize camera with error handling
     */
    async initializeCamera() {
        try {
            console.log('🎥 Initializing camera...');
            
            // Check if browser supports getUserMedia
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera access is not supported in this browser');
            }

            // Get video element
            this.video = document.getElementById('video');
            if (!this.video) {
                throw new Error('Video element not found');
            }

            // Get canvas element
            this.canvas = document.getElementById('canvas');
            if (!this.canvas) {
                throw new Error('Canvas element not found');
            }
            
            this.context = this.canvas.getContext('2d');

            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia(this.config);
            
            // Set video source
            this.video.srcObject = this.stream;
            this.video.play();

            // Wait for video to be ready
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    resolve();
                };
            });

            this.isInitialized = true;
            console.log('✅ Camera initialized successfully');
            
            // Hide overlay
            const overlay = document.getElementById('camera-overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }

            return { success: true };

        } catch (error) {
            console.error('❌ Camera initialization failed:', error);
            this.handleCameraError(error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Stop camera and release resources
     */
    stopCamera() {
        try {
            if (this.stream) {
                this.stream.getTracks().forEach(track => {
                    track.stop();
                });
                this.stream = null;
            }

            if (this.video) {
                this.video.srcObject = null;
            }

            this.isInitialized = false;
            console.log('🛑 Camera stopped');

            // Show overlay
            const overlay = document.getElementById('camera-overlay');
            if (overlay) {
                overlay.style.display = 'flex';
            }

        } catch (error) {
            console.error('Error stopping camera:', error);
        }
    }

    /**
     * Capture photo from video stream
     */
    capturePhoto() {
        if (!this.isInitialized || !this.video || !this.canvas) {
            throw new Error('Camera not initialized');
        }

        try {
            // Set canvas dimensions to match video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            // Draw video frame to canvas
            this.context.drawImage(this.video, 0, 0);

            // Get image data as base64
            const dataURL = this.canvas.toDataURL(this.config.captureFormat, this.config.captureQuality);

            console.log('📸 Photo captured successfully');
            
            return {
                success: true,
                dataURL: dataURL,
                width: this.canvas.width,
                height: this.canvas.height
            };

        } catch (error) {
            console.error('❌ Photo capture failed:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get camera info
     */
    getCameraInfo() {
        if (!this.video || !this.isInitialized) {
            return null;
        }

        return {
            width: this.video.videoWidth,
            height: this.video.videoHeight,
            isPlaying: !this.video.paused && !this.video.ended && this.video.readyState > 2
        };
    }

    /**
     * Handle camera errors with user-friendly messages
     */
    handleCameraError(error) {
        let userMessage = 'Camera access failed. ';
        
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            userMessage += 'Please allow camera access and try again.';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            userMessage += 'No camera found. Please connect a camera and try again.';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            userMessage += 'Camera is being used by another application.';
        } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
            userMessage += 'Camera does not meet the required specifications.';
        } else {
            userMessage += error.message || 'Unknown error occurred.';
        }

        // Show error to user
        this.showError(userMessage);
    }

    /**
     * Show error message to user
     */
    showError(message) {
        const overlay = document.getElementById('camera-overlay');
        if (overlay) {
            overlay.innerHTML = `
                <div class="text-white text-center">
                    <i class="fas fa-exclamation-triangle text-6xl mb-4 text-red-400"></i>
                    <p class="text-xl font-semibold mb-2">Camera Error</p>
                    <p class="text-sm">${message}</p>
                </div>
            `;
            overlay.style.display = 'flex';
        }
    }

    /**
     * Check if camera is supported
     */
    static isCameraSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    /**
     * Request camera permissions
     */
    static async requestPermissions() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch (error) {
            return false;
        }
    }
}

/**
 * Face Detection Utilities
 */
class FaceDetector {
    constructor() {
        this.isDetecting = false;
        this.detectionCallback = null;
        this.detectionInterval = null;
    }

    /**
     * Start face detection
     */
    startDetection(video, callback) {
        if (!video || this.isDetecting) {
            return;
        }

        this.isDetecting = true;
        this.detectionCallback = callback;

        // Simple face detection simulation (in a real app, you'd use a proper face detection library)
        this.detectionInterval = setInterval(() => {
            if (video.videoWidth > 0 && video.videoHeight > 0) {
                // Simulate face detection
                const faceDetected = Math.random() > 0.3; // 70% chance of detection
                
                if (faceDetected && this.detectionCallback) {
                    const fakeRect = {
                        x: video.videoWidth * 0.25,
                        y: video.videoHeight * 0.2,
                        width: video.videoWidth * 0.5,
                        height: video.videoHeight * 0.6,
                        confidence: 0.8 + Math.random() * 0.2
                    };
                    
                    this.detectionCallback(fakeRect);
                }
            }
        }, 1000); // Check every second
    }

    /**
     * Stop face detection
     */
    stopDetection() {
        this.isDetecting = false;
        this.detectionCallback = null;
        
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
    }
}

/**
 * Global camera manager instance
 */
let cameraManager = null;
let faceDetector = null;

/**
 * Initialize camera system
 */
function initializeCamera() {
    console.log('🚀 Initializing camera system...');
    
    // Check browser support
    if (!CameraManager.isCameraSupported()) {
        console.error('❌ Camera not supported');
        alert('Camera is not supported in this browser. Please use a modern browser like Chrome, Firefox, or Safari.');
        return;
    }

    // Create camera manager
    cameraManager = new CameraManager();
    faceDetector = new FaceDetector();

    // Set up event listeners
    setupCameraControls();
    
    console.log('✅ Camera system ready');
}

/**
 * Set up camera control event listeners
 */
function setupCameraControls() {
    const startBtn = document.getElementById('start-camera');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const captureBtn = document.getElementById('capture-photo');
    const capturePhotoBtn = document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-photo');
    const retryBtn = document.getElementById('retry-btn');

    // Start camera buttons
    [startBtn, startCameraBtn].forEach(btn => {
        if (btn) {
            btn.addEventListener('click', handleStartCamera);
        }
    });

    // Capture buttons
    [captureBtn, capturePhotoBtn].forEach(btn => {
        if (btn) {
            btn.addEventListener('click', handleCapturePhoto);
        }
    });

    // Retake/retry buttons
    [retakeBtn, retryBtn].forEach(btn => {
        if (btn) {
            btn.addEventListener('click', handleRetakePhoto);
        }
    });
}

/**
 * Handle start camera button click
 */
async function handleStartCamera() {
    const startBtn = document.getElementById('start-camera') || document.getElementById('start-camera-btn');
    const captureBtn = document.getElementById('capture-photo') || document.getElementById('capture-btn');
    
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Starting...';
    }

    try {
        const result = await cameraManager.initializeCamera();
        
        if (result.success) {
            // Hide start button, show capture button
            if (startBtn) {
                startBtn.classList.add('hidden');
            }
            if (captureBtn) {
                captureBtn.classList.remove('hidden');
            }

            // Start face detection
            const video = document.getElementById('video');
            if (video) {
                faceDetector.startDetection(video, handleFaceDetected);
            }

            console.log('✅ Camera started successfully');
        } else {
            throw new Error(result.error);
        }

    } catch (error) {
        console.error('❌ Failed to start camera:', error);
        
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-video mr-2"></i>Start Camera';
        }
        
        alert('Failed to start camera: ' + error.message);
    }
}

/**
 * Handle capture photo button click
 */
function handleCapturePhoto() {
    const captureBtn = document.getElementById('capture-photo') || document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-photo') || document.getElementById('retry-btn');
    const previewContainer = document.getElementById('photo-preview');
    const previewImage = document.getElementById('previewImage') || document.getElementById('captured-image');
    const photoDataInput = document.getElementById('photo_data');
    const submitBtn = document.getElementById('submit-btn');

    try {
        const result = cameraManager.capturePhoto();
        
        if (result.success) {
            // Hide capture button, show retake button
            if (captureBtn) {
                captureBtn.classList.add('hidden');
            }
            if (retakeBtn) {
                retakeBtn.classList.remove('hidden');
            }

            // Show preview
            if (previewImage) {
                previewImage.src = result.dataURL;
            }
            if (previewContainer) {
                previewContainer.classList.remove('hidden');
            }

            // Store photo data
            if (photoDataInput) {
                photoDataInput.value = result.dataURL;
            }

            // Enable submit button
            if (submitBtn) {
                submitBtn.disabled = false;
            }

            // Stop face detection
            faceDetector.stopDetection();

            console.log('📸 Photo captured and preview shown');

        } else {
            throw new Error(result.error);
        }

    } catch (error) {
        console.error('❌ Failed to capture photo:', error);
        alert('Failed to capture photo: ' + error.message);
    }
}

/**
 * Handle retake photo button click
 */
function handleRetakePhoto() {
    const captureBtn = document.getElementById('capture-photo') || document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-photo') || document.getElementById('retry-btn');
    const previewContainer = document.getElementById('photo-preview');
    const photoDataInput = document.getElementById('photo_data');
    const submitBtn = document.getElementById('submit-btn');

    // Show capture button, hide retake button
    if (captureBtn) {
        captureBtn.classList.remove('hidden');
    }
    if (retakeBtn) {
        retakeBtn.classList.add('hidden');
    }

    // Hide preview
    if (previewContainer) {
        previewContainer.classList.add('hidden');
    }

    // Clear photo data
    if (photoDataInput) {
        photoDataInput.value = '';
    }

    // Disable submit button
    if (submitBtn) {
        submitBtn.disabled = true;
    }

    // Restart face detection
    const video = document.getElementById('video');
    if (video && cameraManager.isInitialized) {
        faceDetector.startDetection(video, handleFaceDetected);
    }

    console.log('🔄 Ready for new photo');
}

/**
 * Handle face detection results
 */
function handleFaceDetected(faceRect) {
    const faceFrame = document.getElementById('face-frame');
    
    if (faceFrame && faceRect) {
        // Position the face frame
        faceFrame.style.left = faceRect.x + 'px';
        faceFrame.style.top = faceRect.y + 'px';
        faceFrame.style.width = faceRect.width + 'px';
        faceFrame.style.height = faceRect.height + 'px';
        faceFrame.classList.remove('hidden');

        // Auto-hide after 3 seconds
        setTimeout(() => {
            if (faceFrame) {
                faceFrame.classList.add('hidden');
            }
        }, 3000);
    }
}

/**
 * Cleanup camera resources when page unloads
 */
window.addEventListener('beforeunload', () => {
    if (cameraManager) {
        cameraManager.stopCamera();
    }
    if (faceDetector) {
        faceDetector.stopDetection();
    }
});

/**
 * Export for use in other modules
 */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CameraManager,
        FaceDetector,
        initializeCamera
    };
}
