# Applying the changes to fix SQLAlchemy deprecation warnings by updating the query method to db.session.get().
import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime, date, timedelta
import json
import csv
import io
from config import Config
from face_recognition_utils import FaceRecognitionUtils
from database import db

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your-fallback-secret-key-here")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config.from_object(Config)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///attendance.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize face recognition utils
face_utils = FaceRecognitionUtils()

# Import models and create tables
from models import Student, Teacher, Attendance

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/student_register', methods=['GET', 'POST'])
def student_register():
    """Student registration with enhanced face recognition"""

    if request.method == 'POST':
        try:
            # Get form data
            roll_number = request.form.get('roll_number', '').strip()
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            course = request.form.get('course', '').strip()
            year = request.form.get('year', '').strip()
            semester = request.form.get('semester', '').strip()
            subject = request.form.get('subject', '').strip()

            # Validation
            if not all([roll_number, name, email, phone, course, year, semester, subject]):
                flash('All fields are required.', 'error')
                return render_template('student_register.html')

            # Check if student already exists
            existing_student = Student.query.filter_by(roll_number=roll_number).first()
            if existing_student:
                flash('Student with this roll number already exists!', 'error')
                return render_template('student_register.html')

            existing_email = Student.query.filter_by(email=email).first()
            if existing_email:
                flash('Student with this email already exists!', 'error')
                return render_template('student_register.html')

            # Handle photo capture
            photo_data = request.form.get('photo_data')
            if not photo_data:
                flash('Please capture a photo using the camera.', 'error')
                return render_template('student_register.html')

            # Process captured photo
            try:
                # Convert base64 to image and extract face encoding
                image_array = face_utils.base64_to_image(photo_data)
                if image_array is None:
                    flash('Invalid photo data. Please try again.', 'error')
                    return render_template('student_register.html')

                # Extract face encoding
                face_encoding = face_utils.extract_face_encoding(image_array)
                if face_encoding is None:
                    flash('No clear face detected. Please ensure good lighting and look directly at the camera.', 'error')
                    return render_template('student_register.html')

                # Save photo
                filename = secure_filename(f"student_{roll_number}.jpg")
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                face_utils.save_image(image_array, photo_path)

                # Create new student
                student = Student(
                    roll_number=roll_number,
                    name=name,
                    email=email,
                    phone=phone,
                    course=course,
                    year=year,
                    semester=semester,
                    subject=subject,
                    photo_path=filename
                )
                student.set_face_encoding(face_encoding)

                db.session.add(student)
                db.session.commit()

                flash('Student registered successfully! You can now login and mark attendance.', 'success')
                return redirect(url_for('index'))

            except Exception as e:
                logger.error(f"Error processing photo: {e}")
                flash('Error processing photo. Please try again.', 'error')
                return render_template('student_register.html')

        except Exception as e:
            logger.error(f"Error during registration: {e}")
            flash('An error occurred during registration. Please try again.', 'error')
            db.session.rollback()

    return render_template('student_register.html')

@app.route('/teacher_register', methods=['GET', 'POST'])
def teacher_register():
    """Teacher registration"""
    if request.method == 'POST':
        try:
            # Get form data
            teacher_id = request.form.get('teacher_id', '').strip()
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            department = request.form.get('department', '').strip()
            subject = request.form.get('subject', '').strip()

            # Validation
            if not all([teacher_id, name, email, phone, department, subject]):
                flash('All fields are required.', 'error')
                return render_template('teacher_register.html')

            # Check if teacher already exists
            existing_teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()
            if existing_teacher:
                flash('Teacher with this ID already exists!', 'error')
                return render_template('teacher_register.html')

            existing_email = Teacher.query.filter_by(email=email).first()
            if existing_email:
                flash('Teacher with this email already exists!', 'error')
                return render_template('teacher_register.html')

            # Handle optional photo upload
            photo = request.files.get('photo')
            filename = None
            face_encoding = None

            if photo and photo.filename:
                # Process uploaded photo
                try:
                    filename = secure_filename(f"teacher_{teacher_id}_{photo.filename}")
                    photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    photo.save(photo_path)

                    # Extract face encoding
                    face_encoding = face_utils.extract_face_encoding(photo_path)
                    if face_encoding is None:
                        logger.warning(f"No face detected in uploaded photo for teacher {teacher_id}")
                        # Don't fail registration, just log warning
                        flash('Photo uploaded but no clear face detected. You can update your photo later.', 'warning')

                except Exception as e:
                    logger.error(f"Error processing photo: {e}")
                    flash('Error processing photo, but registration will continue.', 'warning')
                    filename = None
                    face_encoding = None

            # Create new teacher
            teacher = Teacher(
                teacher_id=teacher_id,
                name=name,
                email=email,
                phone=phone,
                department=department,
                subject=subject,
                photo_path=filename
            )

            if face_encoding is not None:
                teacher.set_face_encoding(face_encoding)

            db.session.add(teacher)
            db.session.commit()

            flash('Teacher registered successfully!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            logger.error(f"Error during registration: {e}")
            flash('An error occurred during registration. Please try again.', 'error')
            db.session.rollback()

    return render_template('teacher_register.html')

@app.route('/login', methods=['POST'])
def login():
    """Login functionality"""
    user_id = request.form.get('user_id', '').strip()
    user_type = request.form.get('user_type')

    if not user_id:
        flash('Please enter your ID.', 'error')
        return redirect(url_for('index'))

    if user_type == 'student':
        user = Student.query.filter_by(roll_number=user_id).first()
        if user:
            session['user_id'] = user.id
            session['user_type'] = 'student'
            session['user_name'] = user.name
            session['roll_number'] = user.roll_number
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('student_dashboard'))
        else:
            flash('Student not found. Please check your roll number.', 'error')

    elif user_type == 'teacher':
        user = Teacher.query.filter_by(teacher_id=user_id).first()
        if user:
            session['user_id'] = user.id
            session['user_type'] = 'teacher'
            session['user_name'] = user.name
            session['teacher_id'] = user.teacher_id
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('teacher_dashboard'))
        else:
            flash('Teacher not found. Please check your teacher ID.', 'error')

    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """Logout functionality"""
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/student_dashboard')
def student_dashboard():
    """Student dashboard with attendance statistics"""
    if 'user_id' not in session or session.get('user_type') != 'student':
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('index'))

    student = db.session.get(Student, session['user_id'])
    if not student:
        flash('Student not found.', 'error')
        return redirect(url_for('index'))

    # Get attendance statistics
    total_attendance = Attendance.query.filter_by(student_id=student.id).count()
    present_days = Attendance.query.filter_by(student_id=student.id, status='Present').count()
    absent_days = total_attendance - present_days
    percentage = round((present_days / total_attendance * 100), 1) if total_attendance > 0 else 0

    # Check if already marked today
    today = date.today()
    today_attendance = Attendance.query.filter_by(
        student_id=student.id,
        date=today
    ).first()

    attendance_stats = {
        'total_days': total_attendance,
        'present_days': present_days,
        'absent_days': absent_days,
        'percentage': percentage,
        'marked_today': today_attendance is not None
    }

    # Get recent attendance (last 7 records)
    recent_attendance = Attendance.query.filter_by(student_id=student.id)\
                                      .order_by(Attendance.created_at.desc())\
                                      .limit(7).all()

    return render_template('student_dashboard.html',
                         student=student,
                         attendance_stats=attendance_stats,
                         recent_attendance=recent_attendance)

@app.route('/teacher_dashboard')
def teacher_dashboard():
    """Teacher dashboard with comprehensive statistics"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('index'))

    teacher = db.session.get(Teacher, session['user_id'])
    if not teacher:
        flash('Teacher not found.', 'error')
        return redirect(url_for('index'))

    # Calculate statistics
    total_students = Student.query.count()
    today = date.today()
    today_present = Attendance.query.filter_by(date=today, status='Present').count()
    today_absent = total_students - today_present if total_students > 0 else 0
    attendance_percentage = round((today_present / total_students * 100), 1) if total_students > 0 else 0

    # Weekly statistics - convert Row objects to serializable format
    week_ago = today - timedelta(days=7)
    weekly_attendance_raw = db.session.query(Attendance.date, db.func.count(Attendance.id))\
                                     .filter(Attendance.date >= week_ago)\
                                     .filter(Attendance.status == 'Present')\
                                     .group_by(Attendance.date)\
                                     .all()
    
    # Convert to JSON-serializable format
    weekly_attendance = [{'date': row[0].strftime('%Y-%m-%d'), 'count': row[1]} for row in weekly_attendance_raw]

    return render_template('teacher_dashboard.html',
                         teacher=teacher,
                         total_students=total_students,
                         today_present=today_present,
                         today_absent=today_absent,
                         attendance_percentage=attendance_percentage,
                         weekly_attendance=weekly_attendance)

@app.route('/mark_attendance')
def mark_attendance():
    """Mark attendance page"""
    if 'user_id' not in session or session.get('user_type') != 'student':
        flash('Please login to mark attendance.', 'error')
        return redirect(url_for('index'))

    # Check if already marked today
    student = db.session.get(Student, session['user_id'])
    today = date.today()
    today_attendance = Attendance.query.filter_by(
        student_id=student.id,
        date=today
    ).first()

    if today_attendance:
        flash('Attendance already marked for today!', 'info')
        return redirect(url_for('student_dashboard'))

    return render_template('mark_attendance.html', student=student)

@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    """Process attendance marking with face recognition"""
    if 'user_id' not in session or session.get('user_type') != 'student':
        return jsonify({'success': False, 'message': 'Please login first.'})

    try:
        student = db.session.get(Student, session['user_id'])
        if not student:
            return jsonify({'success': False, 'message': 'Student not found.'})

        # Check if already marked today
        today = date.today()
        existing_attendance = Attendance.query.filter_by(
            student_id=student.id,
            date=today
        ).first()

        if existing_attendance:
            return jsonify({'success': False, 'message': 'Attendance already marked for today!'})

        # Get photo data - handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            photo_data = data.get('photo_data') or data.get('image')
        else:
            photo_data = request.form.get('photo_data')

        if not photo_data:
            return jsonify({'success': False, 'message': 'No photo data received.'})

        # Convert and process image
        image_array = face_utils.base64_to_image(photo_data)
        if image_array is None:
            return jsonify({'success': False, 'message': 'Invalid photo data.'})

        # Extract face encoding
        current_encoding = face_utils.extract_face_encoding(image_array)
        if current_encoding is None:
            return jsonify({'success': False, 'message': 'No face detected. Please ensure good lighting and look directly at the camera.'})

        # Get stored face encoding
        stored_encoding = student.get_face_encoding()
        if stored_encoding is None:
            return jsonify({'success': False, 'message': 'No registered face found. Please contact administrator.'})

        # Compare faces
        match_result = face_utils.compare_faces(stored_encoding, current_encoding)

        if match_result['is_match']:
            # Mark attendance
            attendance = Attendance(
                student_id=student.id,
                roll_number=student.roll_number,
                student_name=student.name,
                date=today,
                time_in=datetime.now().time(),
                status='Present'
            )

            db.session.add(attendance)
            db.session.commit()

            # Calculate attendance stats
            total_attendance = Attendance.query.filter_by(student_id=student.id).count()
            present_days = Attendance.query.filter_by(student_id=student.id, status='Present').count()
            percentage = round((present_days / total_attendance * 100), 1) if total_attendance > 0 else 100

            return jsonify({
                'success': True,
                'message': f'Attendance marked successfully! Welcome, {student.name}.',
                'confidence': round(match_result['confidence'] * 100, 1),
                'details': {
                    'student_name': student.name,
                    'time_marked': datetime.now().strftime('%H:%M:%S'),
                    'similarity_score': match_result['confidence'],
                    'attendance_percentage': percentage
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Face verification failed. Please try again with better lighting.',
                'confidence': round(match_result['confidence'] * 100, 1)
            })

    except Exception as e:
        logger.error(f"Error processing attendance: {e}")
        return jsonify({'success': False, 'message': 'An error occurred. Please try again.'})

@app.route('/view_attendance')
def view_attendance():
    """View attendance records with search and filter"""
    if 'user_id' not in session:
        flash('Please login to view attendance.', 'error')
        return redirect(url_for('index'))

    # Get search parameters
    roll_number = request.args.get('roll_number', '').strip()
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')

    # Build query
    query = Attendance.query

    # Apply filters
    if roll_number:
        query = query.filter(Attendance.roll_number.like(f'%{roll_number}%'))

    if date_from:
        try:
            from_date = datetime.strptime(date_from, '%Y-%m-%d').date()
            query = query.filter(Attendance.date >= from_date)
        except ValueError:
            flash('Invalid from date format.', 'error')

    if date_to:
        try:
            to_date = datetime.strptime(date_to, '%Y-%m-%d').date()
            query = query.filter(Attendance.date <= to_date)
        except ValueError:
            flash('Invalid to date format.', 'error')

    # For students, show only their records
    if session.get('user_type') == 'student':
        query = query.filter(Attendance.student_id == session['user_id'])

    # Execute query
    attendance_records = query.order_by(Attendance.created_at.desc()).all()

    return render_template('view_attendance.html', attendance_records=attendance_records)

@app.route('/manage_students')
def manage_students():
    """Manage students (teacher only)"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('Access denied. Teachers only.', 'error')
        return redirect(url_for('index'))

    # Get search parameters
    search_query = request.args.get('search', '').strip()
    course_filter = request.args.get('course', '').strip()

    # Build query
    query = Student.query

    if search_query:
        query = query.filter(
            db.or_(
                Student.name.like(f'%{search_query}%'),
                Student.roll_number.like(f'%{search_query}%'),
                Student.email.like(f'%{search_query}%')
            )
        )

    if course_filter:
        query = query.filter(Student.course == course_filter)

    students = query.order_by(Student.created_at.desc()).all()

    # Get all unique courses for filter dropdown
    courses = db.session.query(Student.course).distinct().all()
    courses = [course[0] for course in courses if course[0]]

    return render_template('manage_students.html', students=students, courses=courses)

@app.route('/detect_face_realtime', methods=['POST'])
def detect_face_realtime():
    """Real-time face detection for camera preview"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'face_detected': False, 'message': 'No image data'})

        # Convert base64 image
        image_array = face_utils.base64_to_image(data['image'])
        if image_array is None:
            return jsonify({'face_detected': False, 'message': 'Invalid image data'})

        # Detect faces
        faces = face_utils.detect_faces(image_array)

        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            # Calculate relative coordinates for overlay
            img_height, img_width = image_array.shape[:2]
            face_coords = {
                'x': (x / img_width) * 100,
                'y': (y / img_height) * 100,
                'width': (w / img_width) * 100,
                'height': (h / img_height) * 100
            }

            return jsonify({
                'face_detected': True,
                'message': 'Face detected successfully',
                'face_coords': face_coords
            })
        else:
            return jsonify({
                'face_detected': False,
                'message': 'No face detected - please position your face in the camera'
            })

    except Exception as e:
        logger.error(f"Error in real-time face detection: {e}")
        return jsonify({
            'face_detected': False,
            'message': 'Detection error occurred'
        })

@app.route('/export_attendance')
def export_attendance():
    """Export attendance records to CSV"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('Access denied. Teachers only.', 'error')
        return redirect(url_for('index'))

    # Get all attendance records
    records = Attendance.query.order_by(Attendance.date.desc(), Attendance.time_in.desc()).all()

    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(['Roll Number', 'Student Name', 'Date', 'Time', 'Status'])

    # Write data
    for record in records:
        writer.writerow([
            record.roll_number,
            record.student_name,
            record.date.strftime('%Y-%m-%d'),
            record.time_in.strftime('%H:%M:%S'),
            record.status
        ])

    # Prepare response
    output.seek(0)

    # Create a bytes buffer
    buffer = io.BytesIO()
    buffer.write(output.getvalue().encode('utf-8'))
    buffer.seek(0)

    today = date.today().strftime('%Y%m%d')
    filename = f'attendance_report_{today}.csv'

    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
