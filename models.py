from datetime import datetime
import json
import numpy as np
from database import db

class Student(db.Model):
    __tablename__ = 'student'
    
    id = db.Column(db.Integer, primary_key=True)
    roll_number = db.Column(db.String(20), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(15), nullable=False)
    course = db.Column(db.String(50), nullable=False)
    year = db.Column(db.String(10), nullable=False)
    semester = db.Column(db.String(2), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    face_encoding = db.Column(db.Text, nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    attendances = db.relationship('Attendance', backref='student_rel', lazy=True, cascade='all, delete-orphan')

    def set_face_encoding(self, encoding):
        """Convert numpy array to JSON string for storage"""
        if encoding is not None:
            self.face_encoding = json.dumps(encoding.tolist())

    def get_face_encoding(self):
        """Convert JSON string back to numpy array"""
        if self.face_encoding:
            try:
                return np.array(json.loads(self.face_encoding))
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def __repr__(self):
        return f'<Student {self.roll_number}: {self.name}>'

class Teacher(db.Model):
    __tablename__ = 'teacher'
    
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.String(20), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(15), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    face_encoding = db.Column(db.Text, nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def set_face_encoding(self, encoding):
        """Convert numpy array to JSON string for storage"""
        if encoding is not None:
            self.face_encoding = json.dumps(encoding.tolist())

    def get_face_encoding(self):
        """Convert JSON string back to numpy array"""
        if self.face_encoding:
            try:
                return np.array(json.loads(self.face_encoding))
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def __repr__(self):
        return f'<Teacher {self.teacher_id}: {self.name}>'

class Attendance(db.Model):
    __tablename__ = 'attendance'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False, index=True)
    roll_number = db.Column(db.String(20), nullable=False, index=True)
    student_name = db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date, index=True)
    time_in = db.Column(db.Time, default=datetime.utcnow().time)
    status = db.Column(db.String(20), default='Present')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Constraints
    __table_args__ = (
        db.UniqueConstraint('student_id', 'date', name='unique_student_date_attendance'),
    )

    def __repr__(self):
        return f'<Attendance {self.roll_number} - {self.date} - {self.status}>'
