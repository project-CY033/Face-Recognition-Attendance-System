import os
import logging
import psycopg2
from app import app, db

# Configure logging
logger = logging.getLogger(__name__)

class SupabaseClient:
    """
    Database client wrapper for interacting with PostgreSQL directly
    This is a replacement for the Supabase client that was originally used
    """
    
    def __init__(self):
        """Initialize the database connection"""
        self.database_url = app.config['SQLALCHEMY_DATABASE_URI']
        self.connection = None
        self.use_sqlalchemy_fallback = True  # Default to using SQLAlchemy
        
        # Check for Supabase configuration
        self.supabase_url = os.environ.get('SUPABASE_URL')
        self.supabase_key = os.environ.get('SUPABASE_KEY')
        
        # Only try to connect if using PostgreSQL and we have Supabase credentials
        if self.database_url.startswith('postgresql://') and self.supabase_url and self.supabase_key:
            try:
                self.connection = psycopg2.connect(self.database_url)
                logger.info("Supabase/PostgreSQL connection initialized successfully")
                self.use_sqlalchemy_fallback = False
            except Exception as e:
                logger.error(f"Error initializing Supabase/PostgreSQL connection: {str(e)}")
                logger.info("Falling back to SQLAlchemy for database operations")
                self.connection = None
        else:
            logger.info("Using SQLAlchemy for database operations")
    
    def is_connected(self):
        """Check if the client is connected to the database"""
        return self.connection is not None
    
    def get_client(self):
        """Get the database connection"""
        return self.connection
    
    def upload_face_encoding(self, user_id, face_encoding):
        """
        Store face encoding in the database
        
        Args:
            user_id (int): User ID
            face_encoding (bytes): Face encoding data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Always use SQLAlchemy for this operation for simplicity
            from models import Student
            
            student = Student.query.filter_by(user_id=user_id).first()
            if student:
                student.face_encoding = face_encoding
                student.face_registered = True
                db.session.commit()
                logger.info(f"Face encoding stored successfully for user {user_id}")
                return True
            else:
                logger.error(f"Student with user_id {user_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error storing face encoding: {str(e)}")
            return False
    
    def get_face_encoding(self, user_id):
        """
        Get face encoding from the database
        
        Args:
            user_id (int): User ID
            
        Returns:
            bytes: Face encoding data or None if not found
        """
        try:
            # Always use SQLAlchemy for this operation for simplicity
            from models import Student
            
            student = Student.query.filter_by(user_id=user_id).first()
            if student and student.face_encoding and student.face_registered:
                return student.face_encoding
            else:
                logger.warning(f"Face encoding not found for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting face encoding: {str(e)}")
            return None

# Create a singleton instance
supabase = SupabaseClient()
