import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///attendance.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'
    
    # Face recognition settings - simplified and more reliable
    FACE_RECOGNITION_TOLERANCE = 0.5  # More strict for better accuracy
    MIN_FACE_SIZE = (80, 80)          # Minimum face size for detection
    MAX_FACE_SIZE = (400, 400)        # Maximum face size for detection
    
    # Image quality settings
    MIN_IMAGE_BRIGHTNESS = 40
    MAX_IMAGE_BRIGHTNESS = 220
    MIN_CONTRAST = 20
    
    # Performance settings
    MAX_FILE_SIZE = 5 * 1024 * 1024   # 5MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Security settings
    MAX_LOGIN_ATTEMPTS = 5
    SESSION_TIMEOUT = 3600  # 1 hour
    
    # Application settings
    ITEMS_PER_PAGE = 20
    DEBUG = True
