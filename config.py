import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///attendance.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'

    # Advanced face recognition settings
    FACE_RECOGNITION_TOLERANCE = 0.6        # Primary matching tolerance (standard for face_recognition lib)
    # SECONDARY_TOLERANCE = 0.7             # Secondary/relaxed matching tolerance (less relevant with simplified matching)
    # MIN_CONFIDENCE_SCORE = 0.8            # Minimum confidence for face detection (less relevant with new encoding)

    # Image quality thresholds
    MIN_IMAGE_BRIGHTNESS = 50
    MAX_IMAGE_BRIGHTNESS = 200
    MIN_BLUR_THRESHOLD = 100 # Higher is less blurry
    MIN_CONTRAST_THRESHOLD = 30

    # Anti-spoofing settings (More lenient for development)
    ENABLE_ANTI_SPOOFING = True
    SPOOFING_EDGE_THRESHOLD = 0.1  # Lowered from 0.3 (Used by old custom logic)
    SPOOFING_COLOR_UNIFORMITY_THRESHOLD = 5  # Lowered from 15 (Used by old custom logic)
    SPOOFING_TEXTURE_THRESHOLD = 200  # Lowered from 2000, higher means more texture needed (Used by new basic spoof check)

    # Performance settings (These were for the old OpenCV cascade classifier, less relevant now)
    # FACE_DETECTION_SCALE_FACTOR = 1.03  # More sensitive
    # FACE_DETECTION_MIN_NEIGHBORS = 3  # Lowered for better detection
    FACE_ENCODING_SIZE = (160, 160) # Informational, dlib resnet model uses 150x150 internally after alignment

    # Security settings
    MAX_ATTENDANCE_PER_DAY = 1
    ENABLE_SESSION_TIMEOUT = True
    SESSION_TIMEOUT_MINUTES = 60

    # Logging settings
    ENABLE_DETAILED_LOGGING = True
    LOG_FACE_RECOGNITION_ATTEMPTS = True

    # Development/Testing modes
    DEBUG_MODE = True  # Enable for easier development
    BYPASS_FACE_RECOGNITION = False
    DISABLE_ANTI_SPOOFING = False  # If True, anti-spoofing check is completely skipped
    RELAXED_ANTI_SPOOFING = True  # If True (and DISABLE_ANTI_SPOOFING is False), makes anti-spoofing very lenient
    ULTRA_RELAXED_MODE = True  # New ultra-relaxed mode for testing, makes anti-spoofing very lenient
