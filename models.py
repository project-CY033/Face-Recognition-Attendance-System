from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    roll_number = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    course = db.Column(db.String(50), nullable=False)
    year = db.Column(db.String(10), nullable=False)
    semester = db.Column(db.String(2), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    face_encoding = db.Column(db.Text, nullable=True)  # Store as JSON string
    photo_path = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_face_encoding(self, encoding):
        """Convert numpy array to JSON string for storage"""
        if encoding is not None:
            self.face_encoding = json.dumps(encoding.tolist())

    def get_face_encoding(self):
        """Convert JSON string back to numpy array"""
        if self.face_encoding:
            import numpy as np
            return np.array(json.loads(self.face_encoding))
        return None

class Teacher(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    face_encoding = db.Column(db.Text, nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_face_encoding(self, encoding):
        if encoding is not None:
            self.face_encoding = json.dumps(encoding.tolist())

    def get_face_encoding(self):
        if self.face_encoding:
            import numpy as np
            return np.array(json.loads(self.face_encoding))
        return None

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    roll_number = db.Column(db.String(20), nullable=False)
    student_name = db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    time_in = db.Column(db.Time, default=datetime.utcnow().time)
    status = db.Column(db.String(20), default='Present')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    student = db.relationship('Student', backref=db.backref('attendances', lazy=True))


