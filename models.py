from app import db
from flask_login import UserMixin
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON

# User role enum values
ROLE_STUDENT = 'student'
ROLE_TEACHER = 'teacher'
ROLE_ADMIN = 'admin'

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    student_profile = db.relationship('Student', backref='user', uselist=False, cascade='all, delete-orphan')
    teacher_profile = db.relationship('Teacher', backref='user', uselist=False, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Student(db.Model):
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    roll_number = db.Column(db.String(20), unique=True, nullable=False)
    year = db.Column(db.Integer, nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    face_encoding = db.Column(db.LargeBinary, nullable=True)
    face_registered = db.Column(db.Boolean, default=False)
    
    # Relationships
    attendances = db.relationship('Attendance', backref='student', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Student {self.roll_number}>'

class Teacher(db.Model):
    __tablename__ = 'teachers'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    mobile_number = db.Column(db.String(20), nullable=True)
    
    # Relationships
    subjects = db.relationship('TeacherSubject', backref='teacher', cascade='all, delete-orphan')
    labs = db.relationship('Lab', backref='teacher', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Teacher {self.full_name}>'

class Subject(db.Model):
    __tablename__ = 'subjects'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    
    # Relationships
    teacher_subjects = db.relationship('TeacherSubject', backref='subject', cascade='all, delete-orphan')
    attendances = db.relationship('Attendance', backref='subject', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Subject {self.name} - Semester {self.semester}>'

class TeacherSubject(db.Model):
    __tablename__ = 'teacher_subjects'
    
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=False)
    class_time = db.Column(db.String(50), nullable=True)
    
    def __repr__(self):
        return f'<TeacherSubject {self.teacher_id} - {self.subject_id}>'

class Lab(db.Model):
    __tablename__ = 'labs'
    
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    subject_name = db.Column(db.String(100), nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    days = db.Column(db.String(100), nullable=False)  # Comma-separated days
    lab_time = db.Column(db.String(50), nullable=False)
    
    def __repr__(self):
        return f'<Lab {self.subject_name}>'

class Attendance(db.Model):
    __tablename__ = 'attendance'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    marked_by = db.Column(db.String(20), nullable=False)  # 'face', 'manual', 'teacher'
    modified_by_teacher = db.Column(db.Boolean, default=False)
    modification_note = db.Column(db.String(200), nullable=True)
    
    def __repr__(self):
        return f'<Attendance {self.student_id} - {self.subject_id} - {self.date}>'

class Notification(db.Model):
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Notification {self.id} - {self.user_id}>'

class SemesterSetting(db.Model):
    __tablename__ = 'semester_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    semester = db.Column(db.Integer, nullable=False)
    allow_manual_attendance = db.Column(db.Boolean, default=False)
    additional_settings = db.Column(JSON, nullable=True)
    
    def __repr__(self):
        return f'<SemesterSetting {self.semester}>'
