import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_login import LoginManager
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set up database base class
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
# Use the environment variable DATABASE_URL if available, otherwise fall back to SQLite
# This allows for local development and testing when the PostgreSQL URL is not available
database_url = os.environ.get("DATABASE_URL")

# PostgreSQL variables from environment (for direct connection)
pg_user = os.environ.get("PGUSER")
pg_password = os.environ.get("PGPASSWORD")
pg_host = os.environ.get("PGHOST")
pg_port = os.environ.get("PGPORT")
pg_database = os.environ.get("PGDATABASE")

# If DATABASE_URL is not set but we have PostgreSQL variables, build the URL
if not database_url and pg_user and pg_password and pg_host and pg_port and pg_database:
    database_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
    print("Using PostgreSQL connection parameters to build DATABASE_URL")

# Render.com provides PostgreSQL URLs starting with 'postgres://' rather than 'postgresql://'
# This block ensures compatibility with SQLAlchemy which expects 'postgresql://'
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    print("Converted DATABASE_URL from postgres:// to postgresql://")

# Fall back to SQLite if no valid database URL is provided
if not database_url or (not database_url.startswith("postgresql://") and not database_url.startswith("sqlite://")):
    print("Warning: DATABASE_URL does not appear to be a valid database URL. Using SQLite instead.")
    database_url = "sqlite:///attendance_system.db"

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
print(f"Using database: {app.config['SQLALCHEMY_DATABASE_URI'].split('://')[0]}")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database with the app
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'  # Set the login view

# Import configuration
from config import Config
app.config.from_object(Config)

with app.app_context():
    # Import models
    import models
    from models import Subject
    
    # Create all tables
    db.create_all()
    
    # Seed database with initial data if needed
    def seed_database():
        # Check if we need to seed subjects
        if Subject.query.count() == 0:
            print("No subjects found. Adding example subjects...")
            # Add example subjects for each semester
            example_subjects = [
                # Semester 1
                {"name": "Introduction to Computer Science", "semester": 1},
                {"name": "Mathematics I", "semester": 1},
                {"name": "Physics I", "semester": 1},
                {"name": "English Communication", "semester": 1},
                # Semester 2
                {"name": "Data Structures", "semester": 2},
                {"name": "Mathematics II", "semester": 2},
                {"name": "Physics II", "semester": 2},
                {"name": "Digital Logic", "semester": 2},
                # Semester 3
                {"name": "Object-Oriented Programming", "semester": 3},
                {"name": "Computer Organization", "semester": 3},
                {"name": "Database Systems", "semester": 3},
                {"name": "Discrete Mathematics", "semester": 3},
                # Semester 4
                {"name": "Operating Systems", "semester": 4},
                {"name": "Computer Networks", "semester": 4},
                {"name": "Software Engineering", "semester": 4},
                {"name": "Algorithms Analysis", "semester": 4},
                # Semester 5
                {"name": "Artificial Intelligence", "semester": 5},
                {"name": "Web Development", "semester": 5},
                {"name": "Computer Graphics", "semester": 5},
                {"name": "Theory of Computation", "semester": 5},
                # Semester 6
                {"name": "Machine Learning", "semester": 6},
                {"name": "Mobile App Development", "semester": 6},
                {"name": "Compiler Design", "semester": 6},
                {"name": "Network Security", "semester": 6},
                # Semester 7
                {"name": "Data Mining", "semester": 7},
                {"name": "Cloud Computing", "semester": 7},
                {"name": "Internet of Things", "semester": 7},
                {"name": "Big Data Analytics", "semester": 7},
                # Semester 8
                {"name": "Blockchain Technology", "semester": 8},
                {"name": "Virtual Reality", "semester": 8},
                {"name": "Ethical Hacking", "semester": 8},
                {"name": "Project Management", "semester": 8},
            ]
            
            for subject_data in example_subjects:
                subject = Subject(**subject_data)
                db.session.add(subject)
            
            db.session.commit()
            print(f"Added {len(example_subjects)} example subjects.")
    
    # Call seed function
    seed_database()
    
    # Import and register blueprints
    from routes.auth import auth_bp
    from routes.student import student_bp
    from routes.teacher import teacher_bp
    from routes.admin import admin_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(student_bp, url_prefix='/student')
    app.register_blueprint(teacher_bp, url_prefix='/teacher')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    
    # Setup user loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        return models.User.query.get(int(user_id))
    
    # Import routes (must be after blueprints are registered)
    import routes
