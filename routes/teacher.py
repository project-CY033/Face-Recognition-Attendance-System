import os
import json
import csv
import io
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from models import (
    User, Student, Teacher, Subject, TeacherSubject, Lab, 
    Attendance, Notification, SemesterSetting, ROLE_TEACHER
)
from utils.email_service import send_email
from utils.export import generate_excel, generate_pdf

teacher_bp = Blueprint('teacher', __name__)

def check_teacher_role():
    """Check if the current user is a teacher"""
    if not current_user.is_authenticated or current_user.role != ROLE_TEACHER:
        flash('You do not have permission to access this page', 'danger')
        return False
    return True

@teacher_bp.route('/dashboard')
@login_required
def dashboard():
    """Teacher dashboard"""
    if not check_teacher_role():
        return redirect(url_for('auth.login'))
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        flash('Teacher profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Get assigned subjects
    teacher_subjects = TeacherSubject.query.filter_by(teacher_id=teacher.id).all()
    subjects = [ts.subject for ts in teacher_subjects]
    
    # Get unique semesters from subjects
    semesters = sorted(list(set(subject.semester for subject in subjects)))
    
    # Get labs
    labs = Lab.query.filter_by(teacher_id=teacher.id).all()
    
    # Get recent attendance notifications
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    recent_attendance = Attendance.query.join(Student).join(Subject).filter(
        Subject.id.in_([s.id for s in subjects]),
        Attendance.date >= yesterday
    ).order_by(Attendance.date.desc(), Attendance.time.desc()).limit(10).all()
    
    return render_template(
        'teacher/dashboard.html',
        teacher=teacher,
        teacher_subjects=teacher_subjects,
        subjects=subjects,
        semesters=semesters,
        labs=labs,
        recent_attendance=recent_attendance
    )

@teacher_bp.route('/register', methods=['GET', 'POST'])
@login_required
def register_details():
    """Update teacher profile and subject details"""
    if not check_teacher_role():
        return redirect(url_for('auth.login'))
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        flash('Teacher profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Get all available subjects
    all_subjects = Subject.query.order_by(Subject.semester, Subject.name).all()
    
    # Group subjects by semester
    subjects_by_semester = {}
    for subject in all_subjects:
        if subject.semester not in subjects_by_semester:
            subjects_by_semester[subject.semester] = []
        subjects_by_semester[subject.semester].append(subject)
    
    # Get teacher's current subjects
    current_teacher_subjects = TeacherSubject.query.filter_by(teacher_id=teacher.id).all()
    current_subject_ids = [ts.subject_id for ts in current_teacher_subjects]
    
    # Get teacher's current labs
    current_labs = Lab.query.filter_by(teacher_id=teacher.id).all()
    
    if request.method == 'POST':
        # Update teacher profile
        full_name = request.form.get('full_name')
        mobile_number = request.form.get('mobile_number')
        
        teacher.full_name = full_name
        teacher.mobile_number = mobile_number
        
        # Process subjects
        selected_subjects = request.form.getlist('subjects')
        additional_subject = request.form.get('additional_subject')
        
        # Clear existing teacher-subject relationships
        TeacherSubject.query.filter_by(teacher_id=teacher.id).delete()
        
        # Add selected subjects
        for subject_id in selected_subjects:
            class_time = request.form.get(f'class_time_{subject_id}', '')
            teacher_subject = TeacherSubject(
                teacher_id=teacher.id,
                subject_id=int(subject_id),
                class_time=class_time
            )
            db.session.add(teacher_subject)
        
        # Add additional subject if provided
        if additional_subject and additional_subject.strip():
            semester = request.form.get('additional_subject_semester')
            class_time = request.form.get('additional_subject_time', '')
            
            # Check if subject already exists
            existing_subject = Subject.query.filter_by(
                name=additional_subject,
                semester=int(semester)
            ).first()
            
            if existing_subject:
                # Add existing subject to teacher
                teacher_subject = TeacherSubject(
                    teacher_id=teacher.id,
                    subject_id=existing_subject.id,
                    class_time=class_time
                )
                db.session.add(teacher_subject)
            else:
                # Create new subject and add to teacher
                new_subject = Subject(
                    name=additional_subject,
                    semester=int(semester)
                )
                db.session.add(new_subject)
                db.session.flush()  # Get ID of new subject
                
                teacher_subject = TeacherSubject(
                    teacher_id=teacher.id,
                    subject_id=new_subject.id,
                    class_time=class_time
                )
                db.session.add(teacher_subject)
        
        # Process labs
        has_lab = request.form.get('has_lab') == 'yes'
        
        # Clear existing labs
        Lab.query.filter_by(teacher_id=teacher.id).delete()
        
        if has_lab:
            lab_subject = request.form.get('lab_subject')
            lab_semester = request.form.get('lab_semester')
            lab_days = ','.join(request.form.getlist('lab_days'))
            lab_time = request.form.get('lab_time')
            
            if lab_subject and lab_semester and lab_days and lab_time:
                lab = Lab(
                    teacher_id=teacher.id,
                    subject_name=lab_subject,
                    semester=int(lab_semester),
                    days=lab_days,
                    lab_time=lab_time
                )
                db.session.add(lab)
        
        db.session.commit()
        flash('Teacher profile updated successfully', 'success')
        return redirect(url_for('teacher.dashboard'))
    
    return render_template(
        'teacher/profile.html',
        teacher=teacher,
        subjects_by_semester=subjects_by_semester,
        current_subject_ids=current_subject_ids,
        current_labs=current_labs,
        teacher_subjects=current_teacher_subjects
    )

@teacher_bp.route('/semester/<int:semester_id>')
@login_required
def semester_page(semester_id):
    """Semester-specific page for teacher"""
    if not check_teacher_role():
        return redirect(url_for('auth.login'))
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        flash('Teacher profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Verify teacher has access to this semester
    teacher_subjects = TeacherSubject.query.join(Subject).filter(
        TeacherSubject.teacher_id == teacher.id,
        Subject.semester == semester_id
    ).all()
    
    if not teacher_subjects:
        flash('You do not have access to this semester', 'warning')
        return redirect(url_for('teacher.dashboard'))
    
    # Get subjects for this semester
    subjects = [ts.subject for ts in teacher_subjects]
    
    # Get all students for this semester
    students = Student.query.filter_by(semester=semester_id).order_by(Student.roll_number).all()
    
    # Get semester settings
    semester_setting = SemesterSetting.query.filter_by(semester=semester_id).first()
    
    if not semester_setting:
        # Create default settings if not exist
        semester_setting = SemesterSetting(
            semester=semester_id,
            allow_manual_attendance=False,
            additional_settings={}
        )
        db.session.add(semester_setting)
        db.session.commit()
    
    # Get current date for the form
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    return render_template(
        'teacher/semester_page.html',
        teacher=teacher,
        semester_id=semester_id,
        subjects=subjects,
        students=students,
        semester_setting=semester_setting,
        current_date=current_date
    )

@teacher_bp.route('/attendance-entry/<int:subject_id>')
@login_required
def attendance_entry(subject_id):
    """View and manage attendance for a specific subject"""
    if not check_teacher_role():
        return redirect(url_for('auth.login'))
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        flash('Teacher profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Verify teacher has access to this subject
    teacher_subject = TeacherSubject.query.filter_by(
        teacher_id=teacher.id,
        subject_id=subject_id
    ).first()
    
    if not teacher_subject:
        flash('You do not have access to this subject', 'warning')
        return redirect(url_for('teacher.dashboard'))
    
    subject = Subject.query.get_or_404(subject_id)
    
    # Get all students for this semester
    students = Student.query.filter_by(semester=subject.semester).order_by(Student.roll_number).all()
    
    # Get attendance records for this subject
    # Group by date to show a calendar view
    attendance_records = Attendance.query.filter_by(subject_id=subject_id).order_by(Attendance.date).all()
    
    # Group attendance by date
    attendance_by_date = {}
    all_dates = sorted(list(set(record.date for record in attendance_records)))
    
    for date in all_dates:
        attendance_by_date[date] = [
            record for record in attendance_records if record.date == date
        ]
    
    return render_template(
        'teacher/attendance_entry.html',
        teacher=teacher,
        subject=subject,
        students=students,
        attendance_by_date=attendance_by_date,
        all_dates=all_dates
    )

@teacher_bp.route('/update-attendance', methods=['POST'])
@login_required
def update_attendance():
    """Update attendance record"""
    if not check_teacher_role():
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        return jsonify({'success': False, 'message': 'Teacher profile not found'})
    
    attendance_id = request.form.get('attendance_id')
    action = request.form.get('action')  # 'present', 'absent', 'delete'
    note = request.form.get('note', '')
    
    attendance = Attendance.query.get(attendance_id)
    
    if not attendance:
        return jsonify({'success': False, 'message': 'Attendance record not found'})
    
    # Verify teacher has access to this subject
    teacher_subject = TeacherSubject.query.filter_by(
        teacher_id=teacher.id,
        subject_id=attendance.subject_id
    ).first()
    
    if not teacher_subject:
        return jsonify({'success': False, 'message': 'You do not have access to this subject'})
    
    if action == 'delete':
        # Create notification for the student
        notification = Notification(
            user_id=Student.query.get(attendance.student_id).user_id,
            message=f"Your attendance for {Subject.query.get(attendance.subject_id).name} on {attendance.date} has been removed by {teacher.full_name}."
        )
        db.session.add(notification)
        
        # Delete the attendance record
        db.session.delete(attendance)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Attendance record deleted'})
    else:
        # Mark the attendance as modified
        attendance.modified_by_teacher = True
        attendance.modification_note = note
        db.session.commit()
        
        # Create notification for the student
        notification = Notification(
            user_id=Student.query.get(attendance.student_id).user_id,
            message=f"Your attendance for {Subject.query.get(attendance.subject_id).name} on {attendance.date} has been updated by {teacher.full_name}. Note: {note}"
        )
        db.session.add(notification)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Attendance record updated'})

@teacher_bp.route('/mark-attendance', methods=['POST'])
@login_required
def mark_attendance():
    """Mark attendance for a student manually"""
    if not check_teacher_role():
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        return jsonify({'success': False, 'message': 'Teacher profile not found'})
    
    student_id = request.form.get('student_id')
    subject_id = request.form.get('subject_id')
    date_str = request.form.get('date')
    
    # Verify teacher has access to this subject
    teacher_subject = TeacherSubject.query.filter_by(
        teacher_id=teacher.id,
        subject_id=subject_id
    ).first()
    
    if not teacher_subject:
        return jsonify({'success': False, 'message': 'You do not have access to this subject'})
    
    # Parse date
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid date format'})
    
    # Check if attendance already exists
    existing_attendance = Attendance.query.filter_by(
        student_id=student_id,
        subject_id=subject_id,
        date=date
    ).first()
    
    if existing_attendance:
        return jsonify({'success': False, 'message': 'Attendance record already exists for this date'})
    
    # Create new attendance record
    attendance = Attendance(
        student_id=student_id,
        subject_id=subject_id,
        date=date,
        time=datetime.now().time(),
        marked_by='teacher'
    )
    db.session.add(attendance)
    
    # Create notification for the student
    notification = Notification(
        user_id=Student.query.get(student_id).user_id,
        message=f"Attendance for {Subject.query.get(subject_id).name} on {date} has been marked by {teacher.full_name}."
    )
    db.session.add(notification)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Attendance marked successfully'})

@teacher_bp.route('/export-attendance/<int:subject_id>/<format_type>')
@login_required
def export_attendance(subject_id, format_type):
    """Export attendance records as Excel or PDF"""
    if not check_teacher_role():
        return redirect(url_for('auth.login'))
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        flash('Teacher profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Verify teacher has access to this subject
    teacher_subject = TeacherSubject.query.filter_by(
        teacher_id=teacher.id,
        subject_id=subject_id
    ).first()
    
    if not teacher_subject:
        flash('You do not have access to this subject', 'warning')
        return redirect(url_for('teacher.dashboard'))
    
    subject = Subject.query.get_or_404(subject_id)
    
    # Get all students for this semester
    students = Student.query.filter_by(semester=subject.semester).order_by(Student.roll_number).all()
    
    # Get attendance records for this subject
    attendance_records = Attendance.query.filter_by(subject_id=subject_id).order_by(Attendance.date).all()
    
    # Group attendance by date
    all_dates = sorted(list(set(record.date for record in attendance_records)))
    
    # Create data for export
    export_data = []
    
    # Add header row with dates
    header = ['S.No', 'Name', 'College ID', 'Phone', 'Email']
    for date in all_dates:
        header.append(date.strftime('%Y-%m-%d'))
    
    export_data.append(header)
    
    # Add student rows
    for i, student in enumerate(students, 1):
        row = [
            i,
            student.full_name,
            student.roll_number,
            '',  # Phone (not stored in our model)
            User.query.get(student.user_id).email
        ]
        
        # Add attendance for each date
        for date in all_dates:
            attendance = next(
                (a for a in attendance_records if a.student_id == student.id and a.date == date),
                None
            )
            row.append('Present' if attendance else 'Absent')
        
        export_data.append(row)
    
    # Generate the file based on format_type
    if format_type == 'excel':
        return generate_excel(export_data, f"attendance_{subject.name}")
    elif format_type == 'pdf':
        return generate_pdf(export_data, f"attendance_{subject.name}", subject.name)
    else:
        flash('Invalid export format', 'danger')
        return redirect(url_for('teacher.attendance_entry', subject_id=subject_id))

@teacher_bp.route('/notifications')
@login_required
def notifications():
    """View teacher notifications"""
    if not check_teacher_role():
        return redirect(url_for('auth.login'))
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        flash('Teacher profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Get all notifications
    notifications = Notification.query.filter_by(
        user_id=current_user.id
    ).order_by(Notification.created_at.desc()).all()
    
    return render_template(
        'teacher/notifications.html',
        teacher=teacher,
        notifications=notifications
    )

@teacher_bp.route('/settings/<int:semester_id>', methods=['GET', 'POST'])
@login_required
def settings(semester_id):
    """Manage semester settings"""
    if not check_teacher_role():
        return redirect(url_for('auth.login'))
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        flash('Teacher profile not found', 'danger')
        return redirect(url_for('auth.logout'))
    
    # Verify teacher has access to this semester
    teacher_subjects = TeacherSubject.query.join(Subject).filter(
        TeacherSubject.teacher_id == teacher.id,
        Subject.semester == semester_id
    ).all()
    
    if not teacher_subjects:
        flash('You do not have access to this semester', 'warning')
        return redirect(url_for('teacher.dashboard'))
    
    # Get semester settings
    semester_setting = SemesterSetting.query.filter_by(semester=semester_id).first()
    
    if not semester_setting:
        # Create default settings if not exist
        semester_setting = SemesterSetting(
            semester=semester_id,
            allow_manual_attendance=False,
            additional_settings={}
        )
        db.session.add(semester_setting)
        db.session.commit()
    
    if request.method == 'POST':
        allow_manual = request.form.get('allow_manual_attendance') == 'on'
        
        # Update settings
        semester_setting.allow_manual_attendance = allow_manual
        
        # Get additional settings from form
        additional_settings = {}
        for key in request.form:
            if key.startswith('setting_'):
                setting_name = key[8:]  # Remove 'setting_' prefix
                additional_settings[setting_name] = request.form[key]
        
        semester_setting.additional_settings = additional_settings
        db.session.commit()
        
        flash('Settings updated successfully', 'success')
        return redirect(url_for('teacher.semester_page', semester_id=semester_id))
    
    return render_template(
        'teacher/settings.html',
        teacher=teacher,
        semester_id=semester_id,
        semester_setting=semester_setting
    )

@teacher_bp.route('/send-email', methods=['POST'])
@login_required
def send_email_to_students():
    """Send email to students"""
    if not check_teacher_role():
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    teacher = Teacher.query.filter_by(user_id=current_user.id).first()
    
    if not teacher:
        return jsonify({'success': False, 'message': 'Teacher profile not found'})
    
    student_ids = request.form.getlist('student_ids')
    subject_line = request.form.get('subject')
    message_body = request.form.get('message')
    
    if not student_ids or not subject_line or not message_body:
        return jsonify({'success': False, 'message': 'Missing required fields'})
    
    # Get student emails
    students = Student.query.filter(Student.id.in_(student_ids)).all()
    student_emails = [User.query.get(student.user_id).email for student in students]
    
    if not student_emails:
        return jsonify({'success': False, 'message': 'No valid student emails found'})
    
    try:
        # Send email
        send_email(
            recipient_list=student_emails,
            subject=subject_line,
            body=message_body,
            sender_name=f"{teacher.full_name} (via Attendance System)"
        )
        
        return jsonify({'success': True, 'message': f'Email sent to {len(student_emails)} students'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error sending email: {str(e)}'})
