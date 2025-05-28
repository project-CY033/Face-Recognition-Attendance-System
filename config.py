import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///attendance.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'

    # Advanced face recognition settings
    FACE_RECOGNITION_TOLERANCE = 0.6        # Primary matching tolerance
    SECONDARY_TOLERANCE = 0.7               # Secondary/relaxed matching tolerance
    MIN_CONFIDENCE_SCORE = 0.8              # Minimum confidence for face detection

    # Image quality thresholds
    MIN_IMAGE_BRIGHTNESS = 50
    MAX_IMAGE_BRIGHTNESS = 200
    MIN_BLUR_THRESHOLD = 100
    MIN_CONTRAST_THRESHOLD = 30

    # Anti-spoofing settings
    ENABLE_ANTI_SPOOFING = True
    SPOOFING_EDGE_THRESHOLD = 0.3
    SPOOFING_COLOR_UNIFORMITY_THRESHOLD = 15
    SPOOFING_TEXTURE_THRESHOLD = 2000

    # Performance settings
    FACE_DETECTION_SCALE_FACTOR = 1.05
    FACE_DETECTION_MIN_NEIGHBORS = 6
    FACE_ENCODING_SIZE = (160, 160)

    # Security settings
    MAX_ATTENDANCE_PER_DAY = 1
    ENABLE_SESSION_TIMEOUT = True
    SESSION_TIMEOUT_MINUTES = 60

    # Logging settings
    ENABLE_DETAILED_LOGGING = True
    LOG_FACE_RECOGNITION_ATTEMPTS = True

    # Development/Testing modes
    DEBUG_MODE = False
    BYPASS_FACE_RECOGNITION = False
    DISABLE_ANTI_SPOOFING = False