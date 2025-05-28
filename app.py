from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime, date

from config import Config
from models import db, Student, Teacher, Attendance
from simple_face_recognition import UltraModernFaceRecognition

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Initialize ultra-modern face recognition system
face_utils = UltraModernFaceRecognition()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.before_request
def create_tables():
    """Create database tables and handle migrations"""
    if not hasattr(create_tables, 'called'):
        try:
            # Try to create all tables
            db.create_all()
            
            # Check if semester column exists, if not, recreate tables
            with db.engine.connect() as conn:
                result = conn.execute(db.text("PRAGMA table_info(student)")).fetchall()
                columns = [row[1] for row in result]
                
                if 'semester' not in columns:
                    print("🔄 Migrating database - adding semester column...")
                    
                    # Drop and recreate tables to add new columns
                    db.drop_all()
                    db.create_all()
                    
                    print("✅ Database migration completed!")
                    
        except Exception as e:
            print(f"⚠️ Database migration error: {e}")
            # If there's any error, recreate the database
            db.drop_all()
            db.create_all()
            print("✅ Database recreated!")
            
        create_tables.called = True

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/student_register', methods=['GET', 'POST'])
def student_register():
    """Student registration page"""
    if request.method == 'POST':
        try:
            # Get form data
            roll_number = request.form['roll_number']
            name = request.form['name']
            email = request.form['email']
            phone = request.form['phone']
            course = request.form['course']
            year = request.form['year']
            semester = request.form['semester']
            subject = request.form['subject']

            # Check if student already exists
            existing_student = Student.query.filter_by(roll_number=roll_number).first()
            if existing_student:
                flash('Student with this roll number already exists!', 'error')
                return render_template('student_register.html')

            existing_email = Student.query.filter_by(email=email).first()
            if existing_email:
                flash('Student with this email already exists!', 'error')
                return render_template('student_register.html')

            # Handle photo - only camera capture
            photo_data = request.form.get('photo_data')

            if not photo_data:
                flash('Please capture a photo using the camera.', 'error')
                return render_template('student_register.html')

            # Camera captured photo
            print("📷 Processing camera captured photo...")
            try:
                # Convert base64 to image
                image = face_utils.base64_to_image(photo_data)
                if image is None:
                    flash('Invalid camera photo data. Please try again.', 'error')
                    return render_template('student_register.html')

                # Save camera photo
                filename = secure_filename(f"student_{roll_number}_camera.jpg")
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Convert image to PIL and save
                from PIL import Image as PILImage
                import cv2
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pil_image.save(photo_path, 'JPEG', quality=90)

                # Extract face encoding from image array
                face_encoding = face_utils.extract_face_encoding(image)

            except Exception as e:
                flash(f'Error processing camera photo: {str(e)}', 'error')
                return render_template('student_register.html')

            # Check if face was detected
            if face_encoding is None:
                flash('No face detected in the photo. Please ensure your face is clearly visible and try again.', 'error')
                if filename and os.path.exists(photo_path):
                    os.remove(photo_path)
                return render_template('student_register.html')

            print(f"✅ Face encoding extracted successfully. Shape: {face_encoding.shape}")

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

            flash('Student registered successfully with face recognition setup!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            flash(f'Error during registration: {str(e)}', 'error')
            db.session.rollback()

    return render_template('student_register.html')

@app.route('/teacher_register', methods=['GET', 'POST'])
def teacher_register():
    """Teacher registration page"""
    if request.method == 'POST':
        try:
            # Get form data
            teacher_id = request.form['teacher_id']
            name = request.form['name']
            email = request.form['email']
            phone = request.form['phone']
            department = request.form['department']
            subject = request.form['subject']

            # Check if teacher already exists
            existing_teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()
            if existing_teacher:
                flash('Teacher with this ID already exists!', 'error')
                return render_template('teacher_register.html')

            existing_email = Teacher.query.filter_by(email=email).first()
            if existing_email:
                flash('Teacher with this email already exists!', 'error')
                return render_template('teacher_register.html')

            # Handle photo upload
            photo = request.files['photo']
            if photo and photo.filename:
                filename = secure_filename(f"teacher_{teacher_id}_{photo.filename}")
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                photo.save(photo_path)

                # Extract face encoding
                face_encoding = face_utils.extract_face_encoding(photo_path)
                if face_encoding is None:
                    flash('No face detected in the uploaded photo. Please upload a clear photo with your face visible.', 'error')
                    os.remove(photo_path)
                    return render_template('teacher_register.html')

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
                teacher.set_face_encoding(face_encoding)

                db.session.add(teacher)
                db.session.commit()

                flash('Teacher registered successfully!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Please upload a photo for face recognition.', 'error')

        except Exception as e:
            flash(f'Error during registration: {str(e)}', 'error')
            db.session.rollback()

    return render_template('teacher_register.html')

@app.route('/login', methods=['POST'])
def login():
    """Login functionality"""
    user_id = request.form['user_id']
    user_type = request.form['user_type']

    if user_type == 'student':
        user = Student.query.filter_by(roll_number=user_id).first()
        if user:
            session['user_id'] = user.id
            session['user_type'] = 'student'
            session['user_name'] = user.name
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
    """Student dashboard"""
    if 'user_id' not in session or session.get('user_type') != 'student':
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('index'))

    student = Student.query.get(session['user_id'])
    if not student:
        flash('Student not found.', 'error')
        return redirect(url_for('index'))

    # Get attendance statistics
    total_attendance = Attendance.query.filter_by(student_id=student.id).count()
    present_days = Attendance.query.filter_by(student_id=student.id, status='Present').count()
    absent_days = total_attendance - present_days
    percentage = round((present_days / total_attendance * 100), 1) if total_attendance > 0 else 0

    attendance_stats = {
        'total_days': total_attendance,
        'present_days': present_days,
        'absent_days': absent_days,
        'percentage': percentage
    }

    # Get recent attendance (last 5 records)
    recent_attendance = Attendance.query.filter_by(student_id=student.id)\
                                      .order_by(Attendance.created_at.desc())\
                                      .limit(5).all()

    return render_template('student_dashboard.html',
                         student=student,
                         attendance_stats=attendance_stats,
                         recent_attendance=recent_attendance)

@app.route('/teacher_dashboard')
def teacher_dashboard():
    """Teacher dashboard with statistics"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('index'))

    teacher = Teacher.query.get(session['user_id'])

    # Calculate statistics
    total_students = Student.query.count()
    today = date.today()
    today_present = Attendance.query.filter_by(date=today, status='Present').count()
    today_absent = total_students - today_present if total_students > 0 else 0
    attendance_percentage = round((today_present / total_students * 100), 1) if total_students > 0 else 0

    return render_template('teacher_dashboard.html',
                         teacher=teacher,
                         total_students=total_students,
                         today_present=today_present,
                         today_absent=today_absent,
                         attendance_percentage=attendance_percentage)

@app.route('/mark_attendance')
def mark_attendance():
    """Mark attendance page"""
    if 'user_id' not in session or session.get('user_type') != 'student':
        flash('Please login to mark attendance.', 'error')
        return redirect(url_for('index'))

    return render_template('mark_attendance.html')

@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    """
    Advanced Face Recognition Attendance System
    Features:
    - Multi-stage face detection and verification
    - Anti-spoofing protection
    - Real-time quality assessment
    - Detailed feedback and error handling
    - Security logging and monitoring
    """
    try:
        # 1. Authentication check
        if 'user_id' not in session or session.get('user_type') != 'student':
            return jsonify({
                'success': False, 
                'message': 'Authentication required. Please login as student.',
                'error_code': 'AUTH_REQUIRED'
            })

        # 2. Get logged-in student
        student = Student.query.get(session['user_id'])
        if not student:
            return jsonify({
                'success': False, 
                'message': 'Student account not found. Please re-login.',
                'error_code': 'STUDENT_NOT_FOUND'
            })

        print(f"🎯 Processing attendance for: {student.name} (Roll: {student.roll_number})")

        # 3. Validate request data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False, 
                'message': 'No image data received. Please capture photo again.',
                'error_code': 'NO_IMAGE_DATA'
            })

        # 4. Convert and validate image
        print("📷 Converting base64 image...")
        try:
            image = face_utils.base64_to_image(data['image'])
            if image is None:
                return jsonify({
                    'success': False, 
                    'message': 'Invalid image format. Please try capturing again.',
                    'error_code': 'INVALID_IMAGE'
                })
            
            # Validate image dimensions
            if image.shape[0] < 100 or image.shape[1] < 100:
                return jsonify({
                    'success': False, 
                    'message': 'Image too small. Please move closer to the camera.',
                    'error_code': 'IMAGE_TOO_SMALL'
                })
                
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'Image processing error: {str(e)}. Please try again.',
                'error_code': 'IMAGE_PROCESSING_ERROR'
            })

        # 5. Advanced image quality checks
        print("🔍 Performing image quality assessment...")
        quality_result = face_utils.assess_image_quality(image)
        if not quality_result['is_good_quality']:
            return jsonify({
                'success': False, 
                'message': f'Image quality issue: {quality_result["message"]}. Please try again.',
                'error_code': 'POOR_IMAGE_QUALITY',
                'quality_details': quality_result
            })

        # 6. Multi-stage face detection
        print("🔍 Detecting faces in image...")
        face_detection_result = face_utils.detect_faces_advanced(image)
        
        if not face_detection_result['faces_found']:
            return jsonify({
                'success': False, 
                'message': '😕 No face detected. Please position your face properly in the camera frame.',
                'error_code': 'NO_FACE_DETECTED',
                'suggestions': [
                    'Ensure good lighting on your face',
                    'Look directly at the camera',
                    'Remove any face coverings',
                    'Move closer to the camera'
                ]
            })

        if face_detection_result['multiple_faces']:
            return jsonify({
                'success': False, 
                'message': '👥 Multiple faces detected. Please ensure only you are visible in the frame.',
                'error_code': 'MULTIPLE_FACES',
                'face_count': face_detection_result['face_count']
            })

        # 7. Anti-spoofing protection
        print("🛡️ Performing anti-spoofing checks...")
        spoofing_result = face_utils.detect_spoofing(image)
        if spoofing_result['is_spoofing']:
            return jsonify({
                'success': False, 
                'message': f'🚫 Security alert: {spoofing_result["reason"]}. Please use live camera only.',
                'error_code': 'SPOOFING_DETECTED',
                'security_details': spoofing_result
            })

        # 8. Extract face encoding with confidence score
        print("🧠 Extracting face encoding...")
        encoding_result = face_utils.extract_face_encoding_advanced(image)
        if not encoding_result['success']:
            return jsonify({
                'success': False, 
                'message': f'Face analysis failed: {encoding_result["error"]}',
                'error_code': 'ENCODING_FAILED'
            })

        captured_encoding = encoding_result['encoding']
        confidence_score = encoding_result['confidence']

        print(f"✅ Face encoding extracted. Confidence: {confidence_score:.3f}")

        # 9. Get stored face encoding
        print(f"📋 Retrieving stored face data for {student.roll_number}...")
        stored_encoding = student.get_face_encoding()
        if stored_encoding is None:
            return jsonify({
                'success': False, 
                'message': 'No registered face data found. Please re-register your account.',
                'error_code': 'NO_STORED_ENCODING'
            })

        # 10. Ultra-modern face matching with lightning speed
        print("⚡ Performing lightning-fast face matching...")
        
        # Ultra-modern matching algorithm with optimized tolerance
        tolerance = app.config.get('FACE_RECOGNITION_TOLERANCE', 0.55)  # Optimized for better accuracy
        primary_match, similarity_score = face_utils.lightning_face_matching(stored_encoding, captured_encoding, tolerance=tolerance)
        
        # Secondary verification for borderline cases with relaxed tolerance
        if not primary_match and similarity_score > (tolerance - 0.08):
            print("🔄 Running ultra-fast secondary verification...")
            secondary_match = face_utils.compare_faces_relaxed(stored_encoding, captured_encoding, tolerance=tolerance + 0.08)
            if secondary_match:
                print("✅ Secondary verification passed with ultra-modern algorithm")
                primary_match = True

        print(f"📊 Face matching results:")
        print(f"  - Primary match: {'✅ YES' if primary_match else '❌ NO'}")
        print(f"  - Similarity score: {similarity_score:.3f}")
        print(f"  - Confidence: {confidence_score:.3f}")
        print(f"  - Tolerance: {tolerance}")

        if primary_match and confidence_score > 0.7:  # Optimized threshold for ultra-modern system
            # 11. Check for duplicate attendance
            today = date.today()
            existing_attendance = Attendance.query.filter_by(
                student_id=student.id,
                date=today
            ).first()

            if existing_attendance:
                return jsonify({
                    'success': False, 
                    'message': f'✅ Attendance already marked for today at {existing_attendance.time_in.strftime("%I:%M %p")}',
                    'error_code': 'ALREADY_MARKED',
                    'existing_time': existing_attendance.time_in.strftime("%I:%M %p")
                })

            # 12. Mark attendance with detailed logging
            current_time = datetime.now()
            attendance = Attendance(
                student_id=student.id,
                roll_number=student.roll_number,
                student_name=student.name,
                date=today,
                time_in=current_time.time(),
                status='Present'
            )

            db.session.add(attendance)
            db.session.commit()

            # 13. Success response with analytics
            print(f"🎉 Attendance marked successfully for {student.name}")
            
            # Calculate attendance statistics
            total_days = Attendance.query.filter_by(student_id=student.id).count()
            present_days = Attendance.query.filter_by(student_id=student.id, status='Present').count()
            attendance_percentage = round((present_days / total_days * 100), 1) if total_days > 0 else 100

            return jsonify({
                'success': True,
                'message': f'🎉 Attendance marked successfully for {student.name}!',
                'details': {
                    'student_name': student.name,
                    'roll_number': student.roll_number,
                    'time_marked': current_time.strftime("%I:%M %p"),
                    'date': today.strftime("%B %d, %Y"),
                    'similarity_score': round(similarity_score, 3),
                    'confidence': round(confidence_score, 3),
                    'total_attendance_days': total_days,
                    'attendance_percentage': attendance_percentage
                }
            })

        else:
            # 14. Face recognition failed - detailed feedback
            print(f"❌ Face recognition failed for {student.name}")
            
            failure_reason = "Face does not match registered photo"
            suggestions = [
                "Ensure good lighting on your face",
                "Look directly at the camera",
                "Remove glasses if worn during registration",
                "Try again in better lighting conditions"
            ]
            
            if confidence_score < 0.8:
                failure_reason = "Low face detection confidence"
                suggestions.insert(0, "Move closer to the camera for better face detection")
            
            return jsonify({
                'success': False,
                'message': f'❌ {failure_reason}. Please try again.',
                'error_code': 'FACE_MISMATCH',
                'details': {
                    'similarity_score': round(similarity_score, 3),
                    'confidence': round(confidence_score, 3),
                    'required_similarity': tolerance,
                    'suggestions': suggestions
                }
            })

    except Exception as e:
        print(f"💥 Error in process_attendance: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False, 
            'message': 'System error occurred. Please try again or contact support.',
            'error_code': 'SYSTEM_ERROR',
            'details': str(e) if app.debug else 'Internal server error'
        })

@app.route('/view_attendance')
def view_attendance():
    """View attendance records"""
    if 'user_id' not in session or session.get('user_type') != 'student':
        flash('Please login to view attendance.', 'error')
        return redirect(url_for('index'))

    student = Student.query.get(session['user_id'])

    # Get search parameters
    roll_number = request.args.get('roll_number', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')

    # Build query
    query = Attendance.query

    # Apply roll number filter if provided
    if roll_number:
        query = query.filter(Attendance.roll_number.contains(roll_number))
    elif session.get('user_type') == 'student':
        # If no roll number search and user is student, show only their records
        query = query.filter_by(student_id=student.id)

    # Date filters
    if date_from:
        query = query.filter(Attendance.date >= datetime.strptime(date_from, '%Y-%m-%d').date())
    if date_to:
        query = query.filter(Attendance.date <= datetime.strptime(date_to, '%Y-%m-%d').date())

    # Get results
    attendance_records = query.order_by(Attendance.date.desc(), Attendance.time_in.desc()).all()

    return render_template('view_attendance.html', attendance_records=attendance_records)

@app.route('/database_viewer')
def database_viewer():
    """Database viewer - केवल Teachers के लिए"""
    # Check if user is logged in as teacher
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को Database Access की अनुमति है! कृपया Teacher के रूप में Login करें।', 'error')
        return redirect(url_for('index'))

    # Verify teacher exists
    teacher = Teacher.query.get(session['user_id'])
    if not teacher:
        flash('Teacher account not found. Please login again.', 'error')
        return redirect(url_for('logout'))

    # Get all data from database
    students = Student.query.all()
    teachers = Teacher.query.all()
    attendance_records = Attendance.query.order_by(Attendance.created_at.desc()).limit(50).all()

    # Calculate statistics
    stats = {
        'total_students': Student.query.count(),
        'total_teachers': Teacher.query.count(),
        'total_attendance': Attendance.query.count(),
        'face_encodings': Student.query.filter(Student.face_encoding.isnot(None)).count() +
                         Teacher.query.filter(Teacher.face_encoding.isnot(None)).count()
    }

    return render_template('database_viewer.html',
                         students=students,
                         teachers=teachers,
                         attendance_records=attendance_records,
                         stats=stats,
                         current_teacher=teacher)

@app.route('/edit_attendance/<int:attendance_id>', methods=['GET', 'POST'])
def edit_attendance(attendance_id):
    """Edit attendance record - केवल Teachers के लिए"""
    # Check teacher authentication
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को Attendance Edit करने की अनुमति है!', 'error')
        return redirect(url_for('index'))

    # Get attendance record
    attendance = Attendance.query.get_or_404(attendance_id)

    if request.method == 'POST':
        try:
            # Update attendance data
            attendance.student_name = request.form['student_name']
            attendance.date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
            attendance.time_in = datetime.strptime(request.form['time_in'], '%H:%M').time()
            attendance.status = request.form['status']

            db.session.commit()
            flash(f'Attendance record updated successfully for {attendance.student_name}!', 'success')
            return redirect(url_for('database_viewer'))

        except Exception as e:
            flash(f'Error updating attendance: {str(e)}', 'error')
            db.session.rollback()

    return render_template('edit_attendance.html', attendance=attendance)

@app.route('/delete_attendance/<int:attendance_id>', methods=['POST'])
def delete_attendance(attendance_id):
    """Delete attendance record - केवल Teachers के लिए"""
    # Check teacher authentication
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को Attendance Delete करने की अनुमति है!'})

    try:
        # Get attendance record
        attendance = Attendance.query.get_or_404(attendance_id)
        student_name = attendance.student_name

        # Delete record
        db.session.delete(attendance)
        db.session.commit()

        flash(f'Attendance record deleted successfully for {student_name}!', 'success')
        return jsonify({'success': True, 'message': f'Attendance deleted for {student_name}'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error deleting attendance: {str(e)}'})

@app.route('/bulk_attendance_action', methods=['POST'])
def bulk_attendance_action():
    """Bulk attendance actions - केवल Teachers के लिए"""
    # Check teacher authentication
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को Bulk Actions की अनुमति है!'})

    try:
        data = request.get_json()
        action = data.get('action')
        attendance_ids = data.get('attendance_ids', [])

        if not attendance_ids:
            return jsonify({'success': False, 'message': 'No records selected'})

        if action == 'delete':
            # Bulk delete
            deleted_count = 0
            for att_id in attendance_ids:
                attendance = Attendance.query.get(att_id)
                if attendance:
                    db.session.delete(attendance)
                    deleted_count += 1

            db.session.commit()
            return jsonify({'success': True, 'message': f'{deleted_count} attendance records deleted successfully!'})

        elif action == 'mark_absent':
            # Bulk mark as absent
            updated_count = 0
            for att_id in attendance_ids:
                attendance = Attendance.query.get(att_id)
                if attendance:
                    attendance.status = 'Absent'
                    updated_count += 1

            db.session.commit()
            return jsonify({'success': True, 'message': f'{updated_count} records marked as Absent!'})

        else:
            return jsonify({'success': False, 'message': 'Invalid action'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})



@app.route('/edit_student/<int:student_id>')
def edit_student(student_id):
    """Edit student page"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को अनुमति है!', 'error')
        return redirect(url_for('index'))

    student = Student.query.get_or_404(student_id)
    return render_template('edit_student.html', student=student)

@app.route('/edit_student/<int:student_id>', methods=['POST'])
def update_student(student_id):
    """Update student information"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})

    try:
        student = Student.query.get_or_404(student_id)

        # Update student information
        student.name = request.form.get('name')
        student.email = request.form.get('email')
        student.course = request.form.get('course')
        student.year = request.form.get('year')
        student.semester = request.form.get('semester')
        student.subject = request.form.get('subject')

        db.session.commit()
        flash('Student information updated successfully!', 'success')
        return redirect(url_for('database_viewer'))

    except Exception as e:
        db.session.rollback()
        flash(f'Error updating student: {str(e)}', 'error')
        return redirect(url_for('edit_student', student_id=student_id))

@app.route('/delete_student/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    """Delete student"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})

    try:
        student = Student.query.get_or_404(student_id)

        # Delete related attendance records first
        Attendance.query.filter_by(roll_number=student.roll_number).delete()

        # Delete student
        db.session.delete(student)
        db.session.commit()

        return jsonify({'success': True, 'message': f'Student {student.name} successfully deleted!'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/edit_teacher/<int:teacher_id>')
def edit_teacher(teacher_id):
    """Edit teacher page"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को अनुमति है!', 'error')
        return redirect(url_for('index'))

    teacher = Teacher.query.get_or_404(teacher_id)
    return render_template('edit_teacher.html', teacher=teacher)

@app.route('/edit_teacher/<int:teacher_id>', methods=['POST'])
def update_teacher(teacher_id):
    """Update teacher information"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})

    try:
        teacher = Teacher.query.get_or_404(teacher_id)

        # Update teacher information
        teacher.name = request.form.get('name')
        teacher.email = request.form.get('email')
        teacher.department = request.form.get('department')
        teacher.subject = request.form.get('subject')

        db.session.commit()
        flash('Teacher information updated successfully!', 'success')
        return redirect(url_for('database_viewer'))

    except Exception as e:
        db.session.rollback()
        flash(f'Error updating teacher: {str(e)}', 'error')
        return redirect(url_for('edit_teacher', teacher_id=teacher_id))

@app.route('/delete_teacher/<int:teacher_id>', methods=['POST'])
def delete_teacher(teacher_id):
    """Delete teacher"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})

    try:
        teacher = Teacher.query.get_or_404(teacher_id)

        # Delete teacher
        db.session.delete(teacher)
        db.session.commit()

        return jsonify({'success': True, 'message': f'Teacher {teacher.name} successfully deleted!'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/detect_face_realtime', methods=['POST'])
def detect_face_realtime():
    """Real-time face detection endpoint"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'face_detected': False,
                'message': 'No image data received'
            })

        # Convert base64 image
        image = face_utils.base64_to_image(data['image'])
        if image is None:
            return jsonify({
                'face_detected': False,
                'message': 'Invalid image format'
            })

        # Lightning-fast face detection
        face_result = face_utils.lightning_fast_face_detection(image)
        
        if face_result['faces_found']:
            if face_result['multiple_faces']:
                return jsonify({
                    'face_detected': False,
                    'message': f'Multiple faces detected ({face_result["face_count"]}). Only one person allowed.',
                    'face_count': face_result['face_count']
                })
            
            # Get face coordinates for overlay
            face_coords = None
            if face_result['face_regions']:
                face = face_result['face_regions'][0]
                face_coords = {
                    'x': (face[0] / image.shape[1]) * 100,
                    'y': (face[1] / image.shape[0]) * 100,
                    'width': (face[2] / image.shape[1]) * 100,
                    'height': (face[3] / image.shape[0]) * 100
                }
            
            return jsonify({
                'face_detected': True,
                'message': 'Face detected successfully',
                'face_coords': face_coords,
                'quality': 'good'
            })
        else:
            return jsonify({
                'face_detected': False,
                'message': 'No face detected. Please position your face in the camera.'
            })

    except Exception as e:
        return jsonify({
            'face_detected': False,
            'message': f'Detection error: {str(e)}'
        })

@app.route('/debug_face_test')
def debug_face_test():
    """Debug route to test face recognition without camera"""
    if 'user_id' not in session or session.get('user_type') != 'student':
        return jsonify({'error': 'Please login as student first'})

    student = Student.query.get(session['user_id'])
    if not student:
        return jsonify({'error': 'Student not found'})

    # Get stored face encoding
    stored_encoding = student.get_face_encoding()
    if stored_encoding is None:
        return jsonify({'error': 'No face encoding found for student'})

    # Test with the same encoding (should match)
    is_match = face_utils.compare_faces(stored_encoding, stored_encoding, tolerance=0.8)

    return jsonify({
        'student': student.name,
        'roll_number': student.roll_number,
        'encoding_shape': stored_encoding.shape,
        'self_match': is_match,
        'tolerance': 0.8,
        'debug_mode': app.config.get('DEBUG_MODE', False),
        'bypass_face_recognition': app.config.get('BYPASS_FACE_RECOGNITION', False),
        'disable_anti_spoofing': app.config.get('DISABLE_ANTI_SPOOFING', False)
    })

@app.route('/toggle_production_mode')
def toggle_production_mode():
    """Toggle between debug and production mode (for testing only)"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'error': 'Only teachers can toggle production mode'})

    # Toggle settings
    app.config['BYPASS_FACE_RECOGNITION'] = not app.config.get('BYPASS_FACE_RECOGNITION', False)
    app.config['DISABLE_ANTI_SPOOFING'] = not app.config.get('DISABLE_ANTI_SPOOFING', False)

    return jsonify({
        'message': 'Production mode toggled',
        'bypass_face_recognition': app.config.get('BYPASS_FACE_RECOGNITION', False),
        'disable_anti_spoofing': app.config.get('DISABLE_ANTI_SPOOFING', False),
        'mode': 'DEBUG' if app.config.get('BYPASS_FACE_RECOGNITION', False) else 'OPTIMIZED'
    })

@app.route('/emergency_bypass')
def emergency_bypass():
    """Emergency bypass for testing - enables all bypasses"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'error': 'Only teachers can use emergency bypass'})

    # Enable all bypasses for testing
    app.config['BYPASS_FACE_RECOGNITION'] = True
    app.config['DISABLE_ANTI_SPOOFING'] = True
    app.config['DISABLE_PRIVACY_CHECK'] = True

    return jsonify({
        'message': 'Emergency bypass activated - All security disabled for testing',
        'bypass_face_recognition': True,
        'disable_anti_spoofing': True,
        'disable_privacy_check': True,
        'mode': 'EMERGENCY_BYPASS'
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
