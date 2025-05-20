import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
from models import User, Student, Teacher, ROLE_STUDENT, ROLE_TEACHER, ROLE_ADMIN

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        if current_user.role == ROLE_STUDENT:
            return redirect(url_for('student.dashboard'))
        elif current_user.role == ROLE_TEACHER:
            return redirect(url_for('teacher.dashboard'))
        elif current_user.role == ROLE_ADMIN:
            return redirect(url_for('admin.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter both username and password', 'danger')
            return render_template('login.html')
            
        user = User.query.filter_by(username=username).first()
        
        if not user:
            flash('Username not found', 'danger')
            return render_template('login.html')
            
        if not check_password_hash(user.password_hash, password):
            flash('Invalid password', 'danger')
            return render_template('login.html')
        
        login_user(user)
        
        if user.role == ROLE_STUDENT:
            return redirect(url_for('student.dashboard'))
        elif user.role == ROLE_TEACHER:
            return redirect(url_for('teacher.dashboard'))
        elif user.role == ROLE_ADMIN:
            return redirect(url_for('admin.dashboard'))
    
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration route selection page"""
    if current_user.is_authenticated:
        if current_user.role == ROLE_STUDENT:
            return redirect(url_for('student.dashboard'))
        elif current_user.role == ROLE_TEACHER:
            return redirect(url_for('teacher.dashboard'))
        elif current_user.role == ROLE_ADMIN:
            return redirect(url_for('admin.dashboard'))
    
    return render_template('register.html')

@auth_bp.route('/register/student', methods=['GET', 'POST'])
def register_student():
    """Student registration"""
    if current_user.is_authenticated:
        return redirect(url_for('student.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        roll_number = request.form.get('roll_number')
        year = request.form.get('year')
        semester = request.form.get('semester')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('student/register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('student/register.html')
        
        # Check if roll number already exists
        if Student.query.filter_by(roll_number=roll_number).first():
            flash('Roll number already registered', 'danger')
            return render_template('student/register.html')
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=ROLE_STUDENT
        )
        db.session.add(user)
        db.session.flush()  # Flush to get the user ID
        
        # Create student profile
        student = Student(
            user_id=user.id,
            full_name=full_name,
            roll_number=roll_number,
            year=int(year),
            semester=int(semester),
            face_registered=False
        )
        db.session.add(student)
        db.session.commit()
        
        flash('Registration successful! You can now login.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('student/register.html')

@auth_bp.route('/register/teacher', methods=['GET', 'POST'])
def register_teacher():
    """Teacher registration"""
    if current_user.is_authenticated:
        return redirect(url_for('teacher.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        mobile_number = request.form.get('mobile_number')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('teacher/register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('teacher/register.html')
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=ROLE_TEACHER
        )
        db.session.add(user)
        db.session.flush()  # Flush to get the user ID
        
        # Create teacher profile
        teacher = Teacher(
            user_id=user.id,
            full_name=full_name,
            mobile_number=mobile_number
        )
        db.session.add(teacher)
        db.session.commit()
        
        flash('Registration successful! You can now login.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('teacher/register.html')
