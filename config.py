import os

class Config:
    # Supabase Configuration
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://itcrspdirsdokehaspvb.supabase.co")
  # SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml0Y3JzcGRpcnNkb2tlaGFzcHZiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDcwMjIyNzcsImV4cCI6MjA2MjU5ODI3N30.lOETEfUh3MDgR8Nq4106hfAKO-dd7Jxsar-Rlknqi60")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "enter_anon_public")
    
    # Flask Configuration
    SECRET_KEY = os.environ.get("SESSION_SECRET", "dev_secret_key")
    
    # Application Settings
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Email Configuration
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "True").lower() in ['true', '1', 't']
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "")
    
    # Face Recognition Settings
    FACE_RECOGNITION_MODEL = "hog"  # Options: hog (CPU) or cnn (GPU)
    FACE_RECOGNITION_TOLERANCE = 0.6  # Lower is more strict
