/**
 * Enhanced Attendance System for SmartAttendance
 * Handles attendance marking with face recognition
 */

class AttendanceSystem {
    constructor() {
        this.isProcessing = false;
        this.cameraManager = null;
        this.maxRetries = 3;
        this.currentRetries = 0;
        
        // Initialize camera manager
        this.initializeCameraManager();
        
        // Bind methods
        this.markAttendance = this.markAttendance.bind(this);
        this.handleCameraStart = this.handleCameraStart.bind(this);
        this.handlePhotoCapture = this.handlePhotoCapture.bind(this);
    }

    /**
     * Initialize camera manager
     */
    initializeCameraManager() {
        if (typeof CameraManager !== 'undefined') {
            this.cameraManager = new CameraManager();
        } else {
            console.error('CameraManager not available');
        }
    }

    /**
     * Initialize attendance system
     */
    init() {
        this.setupEventListeners();
        this.checkCameraSupport();
        this.updateUI();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Camera start button
        const startBtn = document.getElementById('start-camera-btn');
        if (startBtn) {
            startBtn.addEventListener('click', this.handleCameraStart);
        }

        // Capture button
        const captureBtn = document.getElementById('capture-btn');
        if (captureBtn) {
            captureBtn.addEventListener('click', this.handlePhotoCapture);
        }

        // Retry button
        const retryBtn = document.getElementById('retry-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', this.handleRetry.bind(this));
        }
    }

    /**
     * Check camera support
     */
    checkCameraSupport() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Camera is not supported in this browser. Please use Chrome, Firefox, or Safari.');
            const startBtn = document.getElementById('start-camera-btn');
            if (startBtn) {
                startBtn.disabled = true;
                startBtn.innerHTML = '<i class="fas fa-times mr-2"></i>Camera Not Supported';
            }
        }
    }

    /**
     * Handle camera start
     */
    async handleCameraStart() {
        const startBtn = document.getElementById('start-camera-btn');
        const captureBtn = document.getElementById('capture-btn');
        const overlay = document.getElementById('camera-overlay');

        try {
            // Update button state
            if (startBtn) {
                startBtn.disabled = true;
                startBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Starting Camera...';
            }

            // Initialize camera
            if (this.cameraManager) {
                const result = await this.cameraManager.initializeCamera();
                
                if (result.success) {
                    // Hide overlay
                    if (overlay) {
                        overlay.style.display = 'none';
                    }

                    // Show/hide buttons
                    if (startBtn) {
                        startBtn.classList.add('hidden');
                    }
                    if (captureBtn) {
                        captureBtn.classList.remove('hidden');
                    }

                    console.log('✅ Camera started successfully');
                } else {
                    throw new Error(result.error || 'Failed to start camera');
                }
            } else {
                throw new Error('Camera manager not available');
            }

        } catch (error) {
            console.error('❌ Camera start failed:', error);
            this.showError('Failed to start camera: ' + error.message);
            
            // Reset button
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.innerHTML = '<i class="fas fa-video mr-2"></i>Start Camera';
            }
        }
    }

    /**
     * Handle photo capture and attendance marking
     */
    async handlePhotoCapture() {
        if (this.isProcessing) {
            return;
        }

        this.isProcessing = true;
        const captureBtn = document.getElementById('capture-btn');
        const retryBtn = document.getElementById('retry-btn');
        const processingIndicator = document.getElementById('processing-indicator');

        try {
            // Update UI
            if (captureBtn) {
                captureBtn.disabled = true;
                captureBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';
            }

            // Show processing indicator
            if (processingIndicator) {
                processingIndicator.classList.remove('hidden');
            }

            // Show loading message
            this.showLoadingMessage();

            // Capture photo
            if (!this.cameraManager) {
                throw new Error('Camera not initialized');
            }

            const captureResult = this.cameraManager.capturePhoto();
            if (!captureResult.success) {
                throw new Error(captureResult.error || 'Failed to capture photo');
            }

            console.log('📸 Photo captured, sending for verification...');

            // Send for attendance marking
            const attendanceResult = await this.markAttendance(captureResult.dataURL);

            if (attendanceResult.success) {
                this.showSuccessMessage(attendanceResult.message, attendanceResult.confidence);
                
                // Hide capture button, show retry button for new attempt
                if (captureBtn) {
                    captureBtn.classList.add('hidden');
                }
                if (retryBtn) {
                    retryBtn.classList.remove('hidden');
                    retryBtn.innerHTML = '<i class="fas fa-home mr-2"></i>Go to Dashboard';
                    retryBtn.onclick = () => window.location.href = '/student_dashboard';
                }

                // Auto-redirect after 5 seconds
                setTimeout(() => {
                    window.location.href = '/student_dashboard';
                }, 5000);

            } else {
                throw new Error(attendanceResult.message || 'Attendance marking failed');
            }

        } catch (error) {
            console.error('❌ Attendance marking failed:', error);
            this.currentRetries++;
            
            this.showErrorMessage(error.message);
            
            // Reset UI for retry
            if (captureBtn) {
                captureBtn.disabled = false;
                captureBtn.innerHTML = '<i class="fas fa-camera mr-2"></i>Capture & Mark Attendance';
            }
            if (retryBtn) {
                retryBtn.classList.remove('hidden');
                retryBtn.innerHTML = '<i class="fas fa-redo mr-2"></i>Try Again';
            }

        } finally {
            this.isProcessing = false;
            
            // Hide processing indicator
            if (processingIndicator) {
                processingIndicator.classList.add('hidden');
            }
        }
    }

    /**
     * Mark attendance with face recognition
     */
    async markAttendance(photoData) {
        try {
            const response = await fetch('/process_attendance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    photo_data: photoData
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('📡 Server response:', result);

            return result;

        } catch (error) {
            console.error('❌ Network error:', error);
            return {
                success: false,
                message: 'Network error: ' + error.message
            };
        }
    }

    /**
     * Handle retry button
     */
    handleRetry() {
        // Reset retry counter
        this.currentRetries = 0;
        
        // Reset UI
        this.hideAllMessages();
        
        const captureBtn = document.getElementById('capture-btn');
        const retryBtn = document.getElementById('retry-btn');
        
        if (captureBtn) {
            captureBtn.classList.remove('hidden');
            captureBtn.disabled = false;
            captureBtn.innerHTML = '<i class="fas fa-camera mr-2"></i>Capture & Mark Attendance';
        }
        
        if (retryBtn) {
            retryBtn.classList.add('hidden');
        }
    }

    /**
     * Show loading message
     */
    showLoadingMessage() {
        this.hideAllMessages();
        const loadingMsg = document.getElementById('loading-message');
        if (loadingMsg) {
            loadingMsg.classList.remove('hidden');
        }
    }

    /**
     * Show success message
     */
    showSuccessMessage(message, confidence) {
        this.hideAllMessages();
        const successMsg = document.getElementById('success-message');
        const confidenceScore = document.getElementById('confidence-score');
        
        if (successMsg) {
            successMsg.classList.remove('hidden');
            
            // Update message content
            const messageText = successMsg.querySelector('p');
            if (messageText) {
                messageText.textContent = message;
            }
        }
        
        if (confidenceScore && confidence) {
            confidenceScore.innerHTML = `
                <i class="fas fa-chart-line mr-1"></i>
                Recognition Confidence: ${confidence}%
            `;
        }
    }

    /**
     * Show error message
     */
    showErrorMessage(message) {
        this.hideAllMessages();
        const errorMsg = document.getElementById('error-message');
        const errorText = document.getElementById('error-text');
        
        if (errorMsg) {
            errorMsg.classList.remove('hidden');
        }
        
        if (errorText) {
            errorText.textContent = message;
        }
    }

    /**
     * Show general error
     */
    showError(message) {
        console.error('Error:', message);
        alert(message);
    }

    /**
     * Hide all status messages
     */
    hideAllMessages() {
        const messages = [
            'loading-message',
            'success-message',
            'error-message'
        ];
        
        messages.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.classList.add('hidden');
            }
        });
    }

    /**
     * Update UI based on current state
     */
    updateUI() {
        // This method can be expanded to handle various UI updates
        console.log('🔄 UI updated');
    }
}

/**
 * Global attendance system instance
 */
let attendanceSystem = null;

/**
 * Initialize attendance system
 */
function initializeAttendanceSystem() {
    console.log('🚀 Initializing attendance system...');
    
    attendanceSystem = new AttendanceSystem();
    attendanceSystem.init();
    
    // Also initialize camera system
    if (typeof initializeCamera === 'function') {
        initializeCamera();
    }
    
    console.log('✅ Attendance system ready');
}

/**
 * Cleanup on page unload
 */
window.addEventListener('beforeunload', () => {
    if (attendanceSystem && attendanceSystem.cameraManager) {
        attendanceSystem.cameraManager.stopCamera();
    }
});

/**
 * Handle page visibility changes (pause/resume camera)
 */
document.addEventListener('visibilitychange', () => {
    if (attendanceSystem && attendanceSystem.cameraManager) {
        if (document.hidden) {
            // Page is hidden, pause camera
            console.log('⏸️ Page hidden, pausing camera');
        } else {
            // Page is visible, resume camera
            console.log('▶️ Page visible, resuming camera');
        }
    }
});

/**
 * Export for use in other modules
 */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AttendanceSystem,
        initializeAttendanceSystem
    };
}
