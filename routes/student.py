import os
import json
import base64
import numpy as np
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from models import Student, Subject, Teacher, TeacherSubject, Attendance, Notification, SemesterSetting, ROLE_STUDENT
from utils.face_recognition import encode_face, recognize_face

student_bp = Blueprint('student', __name__)

def check_student_role():
    """Check if the current user is a student"""
    if not current_user.is_authenticated or current_user.role != ROLE_STUDENT:
        flash('You do not have permission to access this page', 'danger')
        return False
    return True

@student_bp.route('/dashboard')
@login_required
def dashboard():
    """Student dashboard"""
    if not check_student_role():
        return redirect(url_for('auth.login'))
    
    student = Student.query.filter_by(user_id=current_user.id).first()
    
    if not student:
        flash('Student profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Check if face is registered
    if not student.face_registered:
        flash('Please register your face to use the attendance system', 'warning')
        return redirect(url_for('student.register_face'))
    
    # Get subjects for the student's semester
    subjects = Subject.query.filter_by(semester=student.semester).all()
    
    # Get attendance statistics
    attendance_stats = {}
    total_classes = {}
    
    for subject in subjects:
        # Count total attendance for this subject
        attendance_count = Attendance.query.filter_by(
            student_id=student.id,
            subject_id=subject.id
        ).count()
        
        # Assuming total classes is tracked somewhere or approximated
        # For demonstration, let's say we have a fixed number or calculate it
        # In a real system, you would track this more accurately
        total = 30  # Example fixed number
        
        attendance_stats[subject.id] = attendance_count
        total_classes[subject.id] = total
    
    # Get recent notifications
    notifications = Notification.query.filter_by(
        user_id=current_user.id,
        read=False
    ).order_by(Notification.created_at.desc()).limit(5).all()
    
    return render_template(
        'student/dashboard.html',
        student=student,
        subjects=subjects,
        attendance_stats=attendance_stats,
        total_classes=total_classes,
        notifications=notifications
    )

@student_bp.route('/register-face', methods=['GET', 'POST'])
@login_required
def register_face():
    """Register student's face"""
    if not check_student_role():
        return redirect(url_for('auth.login'))
    
    student = Student.query.filter_by(user_id=current_user.id).first()
    
    if not student:
        flash('Student profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    if request.method == 'POST':
        if 'face_image' not in request.form:
            flash('No face image provided', 'danger')
            return redirect(url_for('student.register_face'))
        
        try:
            # Get image data from the form (base64 encoded)
            face_image_b64 = request.form['face_image'].split(',')[1]
            face_image_data = base64.b64decode(face_image_b64)
            
            # Generate face encoding
            face_encoding = encode_face(face_image_data)
            
            if face_encoding is None:
                flash('No face detected in the image. Please try again.', 'danger')
                return redirect(url_for('student.register_face'))
            
            # Store the face encoding
            student.face_encoding = face_encoding.tobytes()
            student.face_registered = True
            db.session.commit()
            
            flash('Face registered successfully', 'success')
            return redirect(url_for('student.dashboard'))
            
        except Exception as e:
            flash(f'Error registering face: {str(e)}', 'danger')
            return redirect(url_for('student.register_face'))
    
    return render_template('student/register_face.html', student=student)

@student_bp.route('/mark-attendance', methods=['GET', 'POST'])
@login_required
def mark_attendance():
    """Mark attendance with face recognition"""
    if not check_student_role():
        return redirect(url_for('auth.login'))
    
    student = Student.query.filter_by(user_id=current_user.id).first()
    
    if not student:
        flash('Student profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    if not student.face_registered:
        flash('Please register your face first', 'warning')
        return redirect(url_for('student.register_face'))
    
    # Get subjects for the student's semester
    subjects = Subject.query.filter_by(semester=student.semester).all()
    
    # Get teachers for each subject
    subject_teachers = {}
    for subject in subjects:
        teachers = Teacher.query.join(TeacherSubject).filter(
            TeacherSubject.subject_id == subject.id
        ).all()
        subject_teachers[subject.id] = teachers
    
    if request.method == 'POST':
        if 'face_image' not in request.form:
            flash('No face image provided', 'danger')
            return redirect(url_for('student.mark_attendance'))
        
        subject_id = request.form.get('subject_id')
        if not subject_id:
            flash('Subject not selected', 'danger')
            return redirect(url_for('student.mark_attendance'))
        
        try:
            # Get image data from the form (base64 encoded)
            face_image_b64 = request.form['face_image'].split(',')[1]
            face_image_data = base64.b64decode(face_image_b64)
            
            # Get the student's stored face encoding
            stored_encoding = np.frombuffer(student.face_encoding, dtype=np.float64)
            
            # Verify face
            is_match = recognize_face(face_image_data, stored_encoding)
            
            if not is_match:
                flash('Face verification failed. Please try again.', 'danger')
                return redirect(url_for('student.mark_attendance'))
            
            # Check if attendance already marked for today
            today = datetime.now().date()
            existing_attendance = Attendance.query.filter_by(
                student_id=student.id,
                subject_id=subject_id,
                date=today
            ).first()
            
            if existing_attendance:
                flash('Attendance already marked for this subject today', 'warning')
                return redirect(url_for('student.dashboard'))
            
            # Mark attendance
            attendance = Attendance(
                student_id=student.id,
                subject_id=subject_id,
                date=today,
                time=datetime.now().time(),
                marked_by='face'
            )
            db.session.add(attendance)
            db.session.commit()
            
            flash('Attendance marked successfully', 'success')
            return redirect(url_for('student.dashboard'))
            
        except Exception as e:
            flash(f'Error marking attendance: {str(e)}', 'danger')
            return redirect(url_for('student.mark_attendance'))
    
    return render_template(
        'student/mark_attendance.html',
        student=student,
        subjects=subjects,
        subject_teachers=subject_teachers
    )

@student_bp.route('/view-attendance')
@login_required
def view_attendance():
    """View student's attendance records"""
    if not check_student_role():
        return redirect(url_for('auth.login'))
    
    student = Student.query.filter_by(user_id=current_user.id).first()
    
    if not student:
        flash('Student profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Get subjects for the student's semester
    subjects = Subject.query.filter_by(semester=student.semester).all()
    
    # Get attendance records
    attendance_records = Attendance.query.filter_by(student_id=student.id).order_by(Attendance.date.desc()).all()
    
    # Organize attendance by subject
    attendance_by_subject = {}
    for subject in subjects:
        subject_attendance = [a for a in attendance_records if a.subject_id == subject.id]
        attendance_by_subject[subject.id] = subject_attendance
    
    return render_template(
        'student/view_attendance.html',
        student=student,
        subjects=subjects,
        attendance_by_subject=attendance_by_subject
    )

@student_bp.route('/check-attendance', methods=['POST'])
@login_required
def check_attendance():
    """Check attendance by roll number (for students)"""
    if not check_student_role():
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    roll_number = request.form.get('roll_number')
    
    student = Student.query.filter_by(roll_number=roll_number).first()
    
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'})
    
    if student.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'You can only check your own attendance'})
    
    # Get subjects for the student's semester
    subjects = Subject.query.filter_by(semester=student.semester).all()
    
    # Get attendance records
    attendance_records = Attendance.query.filter_by(student_id=student.id).all()
    
    # Organize attendance by subject
    attendance_data = {}
    for subject in subjects:
        subject_attendance = [
            {
                'date': a.date.strftime('%Y-%m-%d'),
                'time': a.time.strftime('%H:%M'),
                'marked_by': a.marked_by
            }
            for a in attendance_records if a.subject_id == subject.id
        ]
        attendance_data[subject.name] = subject_attendance
    
    return jsonify({
        'success': True,
        'student': {
            'name': student.full_name,
            'roll_number': student.roll_number,
            'year': student.year,
            'semester': student.semester
        },
        'attendance': attendance_data
    })

@student_bp.route('/mark-manual-attendance', methods=['GET', 'POST'])
@login_required
def mark_manual_attendance():
    """Mark attendance manually (if allowed)"""
    if not check_student_role():
        return redirect(url_for('auth.login'))
    
    student = Student.query.filter_by(user_id=current_user.id).first()
    
    if not student:
        flash('Student profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Check if manual attendance is allowed for this semester
    semester_setting = SemesterSetting.query.filter_by(semester=student.semester).first()
    
    if not semester_setting or not semester_setting.allow_manual_attendance:
        flash('Manual attendance is not allowed for your semester', 'warning')
        return redirect(url_for('student.dashboard'))
    
    # Get subjects for the student's semester
    subjects = Subject.query.filter_by(semester=student.semester).all()
    
    if request.method == 'POST':
        subject_id = request.form.get('subject_id')
        roll_number = request.form.get('roll_number')
        
        if not subject_id or not roll_number:
            flash('Please fill all required fields', 'danger')
            return redirect(url_for('student.mark_manual_attendance'))
        
        # Verify roll number
        if roll_number != student.roll_number:
            flash('Invalid roll number', 'danger')
            return redirect(url_for('student.mark_manual_attendance'))
        
        # Check if attendance already marked for today
        today = datetime.now().date()
        existing_attendance = Attendance.query.filter_by(
            student_id=student.id,
            subject_id=subject_id,
            date=today
        ).first()
        
        if existing_attendance:
            flash('Attendance already marked for this subject today', 'warning')
            return redirect(url_for('student.dashboard'))
        
        # Mark attendance
        attendance = Attendance(
            student_id=student.id,
            subject_id=subject_id,
            date=today,
            time=datetime.now().time(),
            marked_by='manual'
        )
        db.session.add(attendance)
        db.session.commit()
        
        flash('Attendance marked successfully', 'success')
        return redirect(url_for('student.dashboard'))
    
    return render_template(
        'student/mark_manual_attendance.html',
        student=student,
        subjects=subjects
    )
