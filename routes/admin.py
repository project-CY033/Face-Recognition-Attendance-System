from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app import db
from models import (
    User, Student, Teacher, Subject, TeacherSubject, Lab, 
    Attendance, Notification, SemesterSetting, ROLE_ADMIN, ROLE_STUDENT, ROLE_TEACHER
)
from werkzeug.security import generate_password_hash

admin_bp = Blueprint('admin', __name__)

def check_admin_role():
    """Check if the current user is an admin"""
    if not current_user.is_authenticated or current_user.role != ROLE_ADMIN:
        flash('You do not have permission to access this page', 'danger')
        return False
    return True

@admin_bp.route('/dashboard')
@login_required
def dashboard():
    """Admin dashboard"""
    if not check_admin_role():
        return redirect(url_for('auth.login'))
    
    # Get counts for dashboard
    student_count = Student.query.count()
    teacher_count = Teacher.query.count()
    subject_count = Subject.query.count()
    attendance_count = Attendance.query.count()
    
    # Get recent activities
    recent_attendance = Attendance.query.order_by(Attendance.date.desc(), Attendance.time.desc()).limit(10).all()
    
    return render_template(
        'admin/dashboard.html',
        student_count=student_count,
        teacher_count=teacher_count,
        subject_count=subject_count,
        attendance_count=attendance_count,
        recent_attendance=recent_attendance
    )

@admin_bp.route('/students')
@login_required
def students():
    """Manage students"""
    if not check_admin_role():
        return redirect(url_for('auth.login'))
    
    students = Student.query.order_by(Student.roll_number).all()
    
    return render_template(
        'admin/students.html',
        students=students
    )

@admin_bp.route('/teachers')
@login_required
def teachers():
    """Manage teachers"""
    if not check_admin_role():
        return redirect(url_for('auth.login'))
    
    teachers = Teacher.query.order_by(Teacher.full_name).all()
    
    return render_template(
        'admin/teachers.html',
        teachers=teachers
    )

@admin_bp.route('/subjects')
@login_required
def subjects():
    """Manage subjects"""
    if not check_admin_role():
        return redirect(url_for('auth.login'))
    
    subjects = Subject.query.order_by(Subject.semester, Subject.name).all()
    
    return render_template(
        'admin/subjects.html',
        subjects=subjects
    )

@admin_bp.route('/settings')
@login_required
def settings():
    """Manage system settings"""
    if not check_admin_role():
        return redirect(url_for('auth.login'))
    
    # Get all semester settings
    semester_settings = SemesterSetting.query.order_by(SemesterSetting.semester).all()
    
    return render_template(
        'admin/settings.html',
        semester_settings=semester_settings
    )

@admin_bp.route('/add-admin', methods=['GET', 'POST'])
@login_required
def add_admin():
    """Add a new admin user"""
    if not check_admin_role():
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('admin.add_admin'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('admin.add_admin'))
        
        # Create new admin user
        admin = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=ROLE_ADMIN
        )
        db.session.add(admin)
        db.session.commit()
        
        flash('Admin user created successfully', 'success')
        return redirect(url_for('admin.dashboard'))
    
    return render_template('admin/add_admin.html')

@admin_bp.route('/delete-user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    """Delete a user"""
    if not check_admin_role():
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    # Cannot delete yourself
    if user_id == current_user.id:
        return jsonify({'success': False, 'message': 'Cannot delete your own account'})
    
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'})
    
    try:
        # Student and Teacher profiles will be deleted by cascade
        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'User deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error deleting user: {str(e)}'})

@admin_bp.route('/add-subject', methods=['POST'])
@login_required
def add_subject():
    """Add a new subject"""
    if not check_admin_role():
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    name = request.form.get('name')
    semester = request.form.get('semester')
    
    if not name or not semester:
        return jsonify({'success': False, 'message': 'Name and semester are required'})
    
    # Check if subject already exists
    existing_subject = Subject.query.filter_by(
        name=name,
        semester=int(semester)
    ).first()
    
    if existing_subject:
        return jsonify({'success': False, 'message': 'Subject already exists for this semester'})
    
    # Create new subject
    subject = Subject(
        name=name,
        semester=int(semester)
    )
    db.session.add(subject)
    db.session.commit()
    
    return jsonify({
        'success': True, 
        'message': 'Subject added successfully',
        'subject': {
            'id': subject.id,
            'name': subject.name,
            'semester': subject.semester
        }
    })

@admin_bp.route('/update-semester-setting', methods=['POST'])
@login_required
def update_semester_setting():
    """Update semester settings"""
    if not check_admin_role():
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    semester_id = request.form.get('semester_id')
    allow_manual = request.form.get('allow_manual') == 'true'
    
    if not semester_id:
        return jsonify({'success': False, 'message': 'Semester ID is required'})
    
    # Get or create semester setting
    semester_setting = SemesterSetting.query.filter_by(semester=int(semester_id)).first()
    
    if not semester_setting:
        semester_setting = SemesterSetting(
            semester=int(semester_id),
            allow_manual_attendance=allow_manual,
            additional_settings={}
        )
        db.session.add(semester_setting)
    else:
        semester_setting.allow_manual_attendance = allow_manual
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Semester setting updated successfully'
    })
