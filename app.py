from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime, date
from sqlalchemy import inspect as sql_inspect # Renamed to avoid conflict

from config import Config
from models import db, Student, Teacher, Attendance
from simple_face_recognition import SimpleFaceRecognition # Changed class name for clarity

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Initialize face recognition system
face_utils = SimpleFaceRecognition()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.before_request
def create_tables():
    """Create database tables and handle migrations"""
    if not hasattr(create_tables, 'called'):
        try:
            # Create all tables if they don't exist
            db.create_all()
            
            # Check if 'student' table exists and then if 'semester' column exists
            # This is a basic check. For production, use Alembic.
            inspector = sql_inspect(db.engine)
            if inspector.has_table(Student.__tablename__):
                columns = [col['name'] for col in inspector.get_columns(Student.__tablename__)]
                if 'semester' not in columns:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("!!! WARNING: 'semester' column is missing from 'student' table.      !!!")
                    print("!!! This app previously had logic to DROP ALL TABLES and recreate them,!!!")
                    print("!!! which would WIPE ALL YOUR DATA. That has been disabled.          !!!")
                    print("!!! You need to manually add the 'semester' column to the 'student'  !!!")
                    print("!!! table (e.g., using `ALTER TABLE student ADD COLUMN semester VARCHAR;`)")
                    print("!!! or use a proper migration tool like Alembic.                     !!!")
                    print("!!! The application might not function correctly until this is fixed.  !!!")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # The original code here was:
                    # db.drop_all()
                    # db.create_all()
                    # print("✅ Database migration completed! (All tables dropped and recreated)")
                    # This is too destructive for an automatic process.
            
        except Exception as e:
            print(f"⚠️ Database initialization/check error: {e}")
            # Fallback: try to create all tables again if there was a major issue.
            # This won't help with missing columns in existing tables without drop_all.
            try:
                db.create_all()
                print("✅ Database tables ensured/re-attempted.")
            except Exception as e2:
                print(f"⚠️ Persistent database error: {e2}")

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
            photo_data = request.form.get('photo_data') # Base64 string

            if not photo_data:
                flash('Please capture a photo using the camera.', 'error')
                return render_template('student_register.html')

            # Camera captured photo
            print("📷 Processing camera captured photo...")
            filename = None # Initialize filename
            photo_path = None # Initialize photo_path
            try:
                # Convert base64 to image
                image_cv = face_utils.base64_to_image(photo_data) # Returns OpenCV image
                if image_cv is None:
                    flash('Invalid camera photo data. Please try again.', 'error')
                    return render_template('student_register.html')

                # Save camera photo
                filename = secure_filename(f"student_{roll_number}_camera.jpg")
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Save image using OpenCV
                import cv2
                cv2.imwrite(photo_path, image_cv)
                print(f"📸 Photo saved to {photo_path}")

                # Extract face encoding from image array (OpenCV image)
                face_encoding = face_utils.extract_face_encoding(image_cv)


            except Exception as e:
                print(f"Error processing camera photo: {e}")
                flash(f'Error processing camera photo: {str(e)}', 'error')
                return render_template('student_register.html')

            # Check if face was detected and encoding extracted
            if face_encoding is None:
                flash('No face detected or could not extract features. Please ensure your face is clearly visible and try again.', 'error')
                if filename and os.path.exists(photo_path): # If photo was saved but encoding failed
                    try:
                        os.remove(photo_path)
                        print(f"🗑️ Removed photo {photo_path} due to encoding failure.")
                    except OSError as e_os:
                        print(f"Error removing photo {photo_path}: {e_os}")
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
                photo_path=filename # Save relative path for web access
            )
            student.set_face_encoding(face_encoding)

            db.session.add(student)
            db.session.commit()

            flash('Student registered successfully with face recognition setup!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            print(f"Error during registration: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Error during registration: {str(e)}', 'error')
            db.session.rollback()

    return render_template('student_register.html')

@app.route('/teacher_register', methods=['GET', 'POST'])
def teacher_register():
    """Teacher registration page"""
    if request.method == 'POST':
        try:
            # Get form data
            teacher_id_form = request.form['teacher_id'] # Renamed to avoid conflict with model
            name = request.form['name']
            email = request.form['email']
            phone = request.form['phone']
            department = request.form['department']
            subject = request.form['subject']

            # Check if teacher already exists
            existing_teacher = Teacher.query.filter_by(teacher_id=teacher_id_form).first()
            if existing_teacher:
                flash('Teacher with this ID already exists!', 'error')
                return render_template('teacher_register.html')

            existing_email = Teacher.query.filter_by(email=email).first()
            if existing_email:
                flash('Teacher with this email already exists!', 'error')
                return render_template('teacher_register.html')

            # Handle photo upload
            photo_file = request.files['photo'] # Renamed to avoid conflict
            if photo_file and photo_file.filename:
                filename = secure_filename(f"teacher_{teacher_id_form}_{photo_file.filename}")
                photo_path_full = os.path.join(app.config['UPLOAD_FOLDER'], filename) # Full path for saving/processing
                photo_file.save(photo_path_full)
                print(f"📸 Teacher photo saved to {photo_path_full}")

                # Extract face encoding from the saved image file path
                face_encoding = face_utils.extract_face_encoding(photo_path_full)
                if face_encoding is None:
                    flash('No face detected in the uploaded photo. Please upload a clear photo with your face visible.', 'error')
                    try:
                        os.remove(photo_path_full)
                        print(f"🗑️ Removed photo {photo_path_full} due to encoding failure.")
                    except OSError as e_os:
                        print(f"Error removing photo {photo_path_full}: {e_os}")
                    return render_template('teacher_register.html')

                # Create new teacher
                teacher = Teacher(
                    teacher_id=teacher_id_form,
                    name=name,
                    email=email,
                    phone=phone,
                    department=department,
                    subject=subject,
                    photo_path=filename # Save relative path for web access
                )
                teacher.set_face_encoding(face_encoding)

                db.session.add(teacher)
                db.session.commit()

                flash('Teacher registered successfully!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Please upload a photo for face recognition.', 'error')

        except Exception as e:
            print(f"Error during teacher registration: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Error during registration: {str(e)}', 'error')
            db.session.rollback()

    return render_template('teacher_register.html')


@app.route('/login', methods=['POST'])
def login():
    """Login functionality"""
    user_id_form = request.form['user_id'] # Renamed to avoid conflict
    user_type = request.form['user_type']

    if user_type == 'student':
        user = Student.query.filter_by(roll_number=user_id_form).first()
        if user:
            session['user_id'] = user.id
            session['user_type'] = 'student'
            session['user_name'] = user.name
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('student_dashboard'))
        else:
            flash('Student not found. Please check your roll number.', 'error')

    elif user_type == 'teacher':
        user = Teacher.query.filter_by(teacher_id=user_id_form).first()
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

    student_obj = Student.query.get(session['user_id']) # Renamed
    if not student_obj:
        flash('Student not found.', 'error')
        return redirect(url_for('index'))

    # Get attendance statistics
    total_attendance = Attendance.query.filter_by(student_id=student_obj.id).count()
    present_days = Attendance.query.filter_by(student_id=student_obj.id, status='Present').count()
    absent_days = total_attendance - present_days # This might be incorrect if other statuses exist
    percentage = round((present_days / total_attendance * 100), 1) if total_attendance > 0 else 0

    attendance_stats = {
        'total_days': total_attendance,
        'present_days': present_days,
        'absent_days': absent_days,
        'percentage': percentage
    }

    # Get recent attendance (last 5 records)
    recent_attendance = Attendance.query.filter_by(student_id=student_obj.id)\
                                      .order_by(Attendance.created_at.desc())\
                                      .limit(5).all()

    return render_template('student_dashboard.html',
                         student=student_obj,
                         attendance_stats=attendance_stats,
                         recent_attendance=recent_attendance)

@app.route('/teacher_dashboard')
def teacher_dashboard():
    """Teacher dashboard with statistics"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('index'))

    teacher_obj = Teacher.query.get(session['user_id']) # Renamed
    if not teacher_obj: # Added check
        flash('Teacher account not found.', 'error')
        session.clear()
        return redirect(url_for('index'))


    # Calculate statistics
    total_students = Student.query.count()
    today_date = date.today() # Renamed
    today_present = Attendance.query.filter_by(date=today_date, status='Present').count()
    
    # Calculate today_absent more accurately if considering only registered students
    # This calculation can be complex depending on definition of "absent"
    # For simplicity, if total_students participated today:
    # students_attended_today = db.session.query(Attendance.student_id).filter_by(date=today_date).distinct().count()
    # today_absent = total_students - students_attended_today 
    # The original logic was: total_students - today_present, which is fine if all students are expected every day.
    today_absent = total_students - today_present if total_students >= today_present else 0


    attendance_percentage = round((today_present / total_students * 100), 1) if total_students > 0 else 0

    return render_template('teacher_dashboard.html',
                         teacher=teacher_obj,
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
    Face Recognition Attendance System
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
        student_obj = Student.query.get(session['user_id']) # Renamed
        if not student_obj:
            return jsonify({
                'success': False, 
                'message': 'Student account not found. Please re-login.',
                'error_code': 'STUDENT_NOT_FOUND'
            })

        print(f"🎯 Processing attendance for: {student_obj.name} (Roll: {student_obj.roll_number})")

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
            image_cv = face_utils.base64_to_image(data['image']) # OpenCV image
            if image_cv is None:
                return jsonify({
                    'success': False, 
                    'message': 'Invalid image format. Please try capturing again.',
                    'error_code': 'INVALID_IMAGE'
                })
            
            if image_cv.shape[0] < 100 or image_cv.shape[1] < 100: # Basic size check
                return jsonify({
                    'success': False, 
                    'message': 'Image too small. Please move closer to the camera.',
                    'error_code': 'IMAGE_TOO_SMALL'
                })
                
        except Exception as e:
            print(f"Image processing error: {e}")
            return jsonify({
                'success': False, 
                'message': f'Image processing error: {str(e)}. Please try again.',
                'error_code': 'IMAGE_PROCESSING_ERROR'
            })

        # 5. Image quality checks (optional, can be basic or use face_utils)
        print("🔍 Performing image quality assessment...")
        quality_result = face_utils.assess_image_quality(image_cv) # This is now very lenient
        if not quality_result['is_good_quality']:
            # Even if lenient, good to have a check
            # return jsonify({
            #     'success': False, 
            #     'message': f'Image quality issue: {quality_result["message"]}. Please try again.',
            #     'error_code': 'POOR_IMAGE_QUALITY',
            #     'quality_details': quality_result
            # })
            print(f"⚠️ Quality warning: {quality_result['message']}")


        # 6. Face detection
        print("🔍 Detecting faces in image...")
        # This now returns 'dlib_face_locations' among other things
        face_detection_result = face_utils.detect_faces_advanced(image_cv) 
        
        if not face_detection_result['faces_found']:
            return jsonify({
                'success': False, 
                'message': '😕 No face detected. Please position your face properly.',
                'error_code': 'NO_FACE_DETECTED',
                'suggestions': ['Ensure good lighting', 'Look at camera', 'Remove coverings']
            })

        if face_detection_result['multiple_faces']:
            return jsonify({
                'success': False, 
                'message': '👥 Multiple faces detected. Only one person allowed.',
                'error_code': 'MULTIPLE_FACES',
                'face_count': face_detection_result['face_count']
            })

        # 7. Anti-spoofing (simplified)
        print("🛡️ Performing anti-spoofing checks...")
        if not app.config.get('DISABLE_ANTI_SPOOFING', False):
            if app.config.get('ULTRA_RELAXED_MODE', False) or app.config.get('RELAXED_ANTI_SPOOFING', False):
                print("🔧 Using relaxed/ultra-relaxed anti-spoofing mode.")
                # The new detect_spoofing in simple_face_recognition.py handles these flags
                spoofing_result = face_utils.detect_spoofing(image_cv)
                if spoofing_result['is_spoofing']:
                    print(f"⚠️ Relaxed spoofing check failed: {spoofing_result['reason']}")
                    # Decide if to block even in relaxed mode. For now, let's be very lenient.
                    # if spoofing_result['confidence'] < 0.3: # Example: block if very low confidence
                    #    return jsonify(...) 
            else: # Standard anti-spoofing
                spoofing_result = face_utils.detect_spoofing(image_cv)
                if spoofing_result['is_spoofing']:
                    return jsonify({
                        'success': False, 
                        'message': f'🚫 Security alert: {spoofing_result["reason"]}. Live camera only.',
                        'error_code': 'SPOOFING_DETECTED',
                        'security_details': spoofing_result
                    })
        else:
            print("⚠️ Anti-spoofing disabled by config.")


        # 8. Extract face encoding using the detected face location
        print("🧠 Extracting face encoding...")
        # Use the first (and only) detected face's dlib_location
        dlib_location = face_detection_result['dlib_face_locations'][0]
        encoding_result = face_utils.extract_specific_face_encoding(image_cv, dlib_location)

        if not encoding_result['success']:
            return jsonify({
                'success': False, 
                'message': f'Face analysis failed: {encoding_result["error"]}',
                'error_code': 'ENCODING_FAILED'
            })

        captured_encoding = encoding_result['encoding']
        # The new 'confidence' from encoding_result is static (0.95 on success).
        # The actual matching similarity is more important.
        print(f"✅ Face encoding extracted. Static success confidence: {encoding_result['confidence']:.3f}")


        # 9. Get stored face encoding
        print(f"📋 Retrieving stored face data for {student_obj.roll_number}...")
        stored_encoding = student_obj.get_face_encoding()
        if stored_encoding is None:
            return jsonify({
                'success': False, 
                'message': 'No registered face data. Please re-register your account.',
                'error_code': 'NO_STORED_ENCODING'
            })

        # 10. Face matching
        print("⚡ Performing face matching...")
        tolerance = app.config.get('FACE_RECOGNITION_TOLERANCE', 0.6)
        # compare_faces_advanced now returns (is_match, similarity_score)
        # where similarity_score is 1 - distance. Higher is better.
        primary_match, similarity_score = face_utils.compare_faces_advanced(
            stored_encoding, captured_encoding, tolerance=tolerance
        )
        
        print(f"📊 Face matching results:")
        print(f"  - Match Status: {'✅ YES' if primary_match else '❌ NO'}")
        print(f"  - Similarity Score (1-distance): {similarity_score:.3f}")
        print(f"  - Tolerance: {tolerance}")

        # The old confidence_score > 0.7 check might not be needed if encoding_result['confidence'] is static
        # Rely mainly on primary_match which considers the tolerance.
        if primary_match:
            # 11. Check for duplicate attendance
            today_date = date.today() # Renamed
            existing_attendance = Attendance.query.filter_by(
                student_id=student_obj.id,
                date=today_date
            ).first()

            if existing_attendance:
                return jsonify({
                    'success': False, 
                    'message': f'✅ Attendance already marked for today at {existing_attendance.time_in.strftime("%I:%M %p")}',
                    'error_code': 'ALREADY_MARKED',
                    'existing_time': existing_attendance.time_in.strftime("%I:%M %p")
                })

            # 12. Mark attendance
            current_time = datetime.now()
            attendance_record = Attendance( # Renamed
                student_id=student_obj.id,
                roll_number=student_obj.roll_number,
                student_name=student_obj.name,
                date=today_date,
                time_in=current_time.time(),
                status='Present'
            )

            db.session.add(attendance_record)
            db.session.commit()

            print(f"🎉 Attendance marked successfully for {student_obj.name}")
            
            total_days = Attendance.query.filter_by(student_id=student_obj.id).count()
            present_days = Attendance.query.filter_by(student_id=student_obj.id, status='Present').count()
            attendance_percentage = round((present_days / total_days * 100), 1) if total_days > 0 else 100

            return jsonify({
                'success': True,
                'message': f'🎉 Attendance marked for {student_obj.name}!',
                'details': {
                    'student_name': student_obj.name,
                    'roll_number': student_obj.roll_number,
                    'time_marked': current_time.strftime("%I:%M %p"),
                    'date': today_date.strftime("%B %d, %Y"),
                    'similarity_score': round(similarity_score, 3),
                    'confidence': round(encoding_result['confidence'], 3), # Static confidence
                    'total_attendance_days': total_days,
                    'attendance_percentage': attendance_percentage
                }
            })

        else:
            print(f"❌ Face recognition failed for {student_obj.name}")
            failure_reason = "Face does not match registered photo"
            suggestions = [
                "Ensure good lighting on your face",
                "Look directly at the camera",
                "Try again in better lighting conditions"
            ]
            
            return jsonify({
                'success': False,
                'message': f'❌ {failure_reason}. Please try again.',
                'error_code': 'FACE_MISMATCH',
                'details': {
                    'similarity_score': round(similarity_score, 3),
                    'required_similarity_threshold_for_match_at_tolerance': round(1 - tolerance, 3), # distance must be <= tolerance
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
    # Teacher or Student can view. If student, only their own.
    # If teacher, can search by roll_number.
    if 'user_id' not in session:
        flash('Please login to view attendance.', 'error')
        return redirect(url_for('index'))

    query = Attendance.query
    user_type = session.get('user_type')

    roll_number_search = request.args.get('roll_number', '') # Renamed
    date_from_str = request.args.get('date_from', '') # Renamed
    date_to_str = request.args.get('date_to', '') # Renamed

    if user_type == 'student':
        student_obj = Student.query.get(session['user_id'])
        if not student_obj:
            flash('Student not found.', 'error')
            return redirect(url_for('index'))
        # Students can only see their own, roll_number_search is ignored for them
        query = query.filter_by(student_id=student_obj.id)
    elif user_type == 'teacher':
        if roll_number_search:
            query = query.filter(Attendance.roll_number.contains(roll_number_search))
    else: # Should not happen
        flash('Invalid user type.', 'error')
        return redirect(url_for('index'))
    
    try:
        if date_from_str:
            date_from_obj = datetime.strptime(date_from_str, '%Y-%m-%d').date() # Renamed
            query = query.filter(Attendance.date >= date_from_obj)
        if date_to_str:
            date_to_obj = datetime.strptime(date_to_str, '%Y-%m-%d').date() # Renamed
            query = query.filter(Attendance.date <= date_to_obj)
    except ValueError:
        flash('Invalid date format. Please use YYYY-MM-DD.', 'error')
        # Show unfiltered results or return early
        # For now, proceed with potentially partially filtered query

    attendance_records = query.order_by(Attendance.date.desc(), Attendance.time_in.desc()).all()
    return render_template('view_attendance.html', attendance_records=attendance_records)


@app.route('/database_viewer')
def database_viewer():
    """Database viewer - केवल Teachers के लिए"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को Database Access की अनुमति है! कृपया Teacher के रूप में Login करें।', 'error')
        return redirect(url_for('index'))

    teacher_obj = Teacher.query.get(session['user_id']) # Renamed
    if not teacher_obj:
        flash('Teacher account not found. Please login again.', 'error')
        return redirect(url_for('logout'))

    students_list = Student.query.all() # Renamed
    teachers_list = Teacher.query.all() # Renamed
    # Limit attendance records for performance
    attendance_records_list = Attendance.query.order_by(Attendance.created_at.desc()).limit(100).all() # Renamed & Limited

    stats = {
        'total_students': Student.query.count(),
        'total_teachers': Teacher.query.count(),
        'total_attendance': Attendance.query.count(),
        'face_encodings': Student.query.filter(Student.face_encoding.isnot(None)).count() + \
                         Teacher.query.filter(Teacher.face_encoding.isnot(None)).count()
    }

    return render_template('database_viewer.html',
                         students=students_list,
                         teachers=teachers_list,
                         attendance_records=attendance_records_list,
                         stats=stats,
                         current_teacher=teacher_obj)


@app.route('/edit_attendance/<int:attendance_id>', methods=['GET', 'POST'])
def edit_attendance(attendance_id):
    """Edit attendance record - केवल Teachers के लिए"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को Attendance Edit करने की अनुमति है!', 'error')
        return redirect(url_for('index'))

    attendance_record = Attendance.query.get_or_404(attendance_id) # Renamed

    if request.method == 'POST':
        try:
            attendance_record.student_name = request.form['student_name']
            attendance_record.date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
            attendance_record.time_in = datetime.strptime(request.form['time_in'], '%H:%M').time()
            attendance_record.status = request.form['status']

            db.session.commit()
            flash(f'Attendance record updated successfully for {attendance_record.student_name}!', 'success')
            return redirect(url_for('database_viewer'))

        except Exception as e:
            flash(f'Error updating attendance: {str(e)}', 'error')
            db.session.rollback()

    return render_template('edit_attendance.html', attendance=attendance_record)


@app.route('/delete_attendance/<int:attendance_id>', methods=['POST'])
def delete_attendance(attendance_id):
    """Delete attendance record - केवल Teachers के लिए"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को Attendance Delete करने की अनुमति है!'})

    try:
        attendance_record = Attendance.query.get_or_404(attendance_id) # Renamed
        student_name_deleted = attendance_record.student_name # Renamed

        db.session.delete(attendance_record)
        db.session.commit()

        # flash(f'Attendance record deleted successfully for {student_name_deleted}!', 'success') # flash won't show on jsonify
        return jsonify({'success': True, 'message': f'Attendance deleted for {student_name_deleted}'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error deleting attendance: {str(e)}'})


@app.route('/bulk_attendance_action', methods=['POST'])
def bulk_attendance_action():
    """Bulk attendance actions - केवल Teachers के लिए"""
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को Bulk Actions की अनुमति है!'})

    try:
        data = request.get_json()
        action = data.get('action')
        attendance_ids = data.get('attendance_ids', [])

        if not attendance_ids:
            return jsonify({'success': False, 'message': 'No records selected'})

        count = 0
        if action == 'delete':
            for att_id in attendance_ids:
                attendance_record = Attendance.query.get(att_id) # Renamed
                if attendance_record:
                    db.session.delete(attendance_record)
                    count += 1
            message = f'{count} attendance records deleted successfully!'
        elif action == 'mark_absent':
            for att_id in attendance_ids:
                attendance_record = Attendance.query.get(att_id) # Renamed
                if attendance_record:
                    attendance_record.status = 'Absent'
                    count += 1
            message = f'{count} records marked as Absent!'
        else:
            return jsonify({'success': False, 'message': 'Invalid action'})

        if count > 0:
            db.session.commit()
        return jsonify({'success': True, 'message': message})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/edit_student/<int:student_id_param>', methods=['GET', 'POST']) # Renamed param
def edit_student(student_id_param):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को अनुमति है!', 'error')
        return redirect(url_for('index'))

    student_obj = Student.query.get_or_404(student_id_param) # Renamed

    if request.method == 'POST':
        try:
            student_obj.name = request.form.get('name', student_obj.name)
            student_obj.email = request.form.get('email', student_obj.email)
            # Prevent changing roll_number easily, it's a key identifier
            # student_obj.roll_number = request.form.get('roll_number', student_obj.roll_number)
            student_obj.course = request.form.get('course', student_obj.course)
            student_obj.year = request.form.get('year', student_obj.year)
            student_obj.semester = request.form.get('semester', student_obj.semester)
            student_obj.subject = request.form.get('subject', student_obj.subject)
            # Photo and face encoding updates would require more complex logic here
            db.session.commit()
            flash('Student information updated successfully!', 'success')
            return redirect(url_for('database_viewer'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating student: {str(e)}', 'error')
            # Stay on edit page if error
            return render_template('edit_student.html', student=student_obj)
            
    return render_template('edit_student.html', student=student_obj)


@app.route('/delete_student/<int:student_id_param>', methods=['POST']) # Renamed param
def delete_student(student_id_param):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})

    try:
        student_obj = Student.query.get_or_404(student_id_param) # Renamed
        student_name_deleted = student_obj.name # Renamed

        # Delete related attendance records first
        Attendance.query.filter_by(student_id=student_obj.id).delete() # Use student_id for FK

        # Delete student photo if exists
        if student_obj.photo_path:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], student_obj.photo_path))
            except OSError as e:
                print(f"Error deleting student photo file {student_obj.photo_path}: {e}")


        db.session.delete(student_obj)
        db.session.commit()

        return jsonify({'success': True, 'message': f'Student {student_name_deleted} successfully deleted!'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/edit_teacher/<int:teacher_id_param>', methods=['GET', 'POST']) # Renamed param
def edit_teacher(teacher_id_param):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को अनुमति है!', 'error')
        return redirect(url_for('index'))

    teacher_obj = Teacher.query.get_or_404(teacher_id_param) # Renamed
    if request.method == 'POST':
        try:
            teacher_obj.name = request.form.get('name', teacher_obj.name)
            teacher_obj.email = request.form.get('email', teacher_obj.email)
            teacher_obj.department = request.form.get('department', teacher_obj.department)
            teacher_obj.subject = request.form.get('subject', teacher_obj.subject)
            db.session.commit()
            flash('Teacher information updated successfully!', 'success')
            return redirect(url_for('database_viewer'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating teacher: {str(e)}', 'error')
            return render_template('edit_teacher.html', teacher=teacher_obj)

    return render_template('edit_teacher.html', teacher=teacher_obj)


@app.route('/delete_teacher/<int:teacher_id_param>', methods=['POST']) # Renamed param
def delete_teacher(teacher_id_param):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})

    try:
        teacher_obj = Teacher.query.get_or_404(teacher_id_param) # Renamed
        teacher_name_deleted = teacher_obj.name # Renamed

        # Delete teacher photo if exists
        if teacher_obj.photo_path:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], teacher_obj.photo_path))
            except OSError as e:
                print(f"Error deleting teacher photo file {teacher_obj.photo_path}: {e}")

        db.session.delete(teacher_obj)
        db.session.commit()

        return jsonify({'success': True, 'message': f'Teacher {teacher_name_deleted} successfully deleted!'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/detect_face_realtime', methods=['POST'])
def detect_face_realtime():
    """Real-time face detection endpoint for UI feedback"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'face_detected': False, 'message': 'No image data'})

        image_cv = face_utils.base64_to_image(data['image']) # OpenCV image
        if image_cv is None:
            return jsonify({'face_detected': False, 'message': 'Invalid image format'})

        # Use the new simplified detection
        detection_result = face_utils.detect_faces_advanced(image_cv)
        
        if detection_result['faces_found']:
            if detection_result['multiple_faces']:
                return jsonify({
                    'face_detected': False, # Technically detected, but not valid for single user
                    'message': f'Multiple faces ({detection_result["face_count"]}). One person please.',
                    'face_count': detection_result['face_count']
                })
            
            # Get face coordinates for overlay from dlib_face_locations
            # dlib_face_locations are (top, right, bottom, left)
            face_coords = None
            if detection_result['dlib_face_locations']:
                top, right, bottom, left = detection_result['dlib_face_locations'][0]
                img_h, img_w = image_cv.shape[:2]
                
                face_coords = {
                    'x': (left / img_w) * 100,
                    'y': (top / img_h) * 100,
                    'width': ((right - left) / img_w) * 100,
                    'height': ((bottom - top) / img_h) * 100
                }
            
            return jsonify({
                'face_detected': True,
                'message': 'Face detected!',
                'face_coords': face_coords,
                'quality': 'good' # Assuming quality is good if detected by new method
            })
        else:
            return jsonify({
                'face_detected': False,
                'message': 'No face detected. Position face in camera.'
            })

    except Exception as e:
        print(f"Error in /detect_face_realtime: {e}")
        return jsonify({'face_detected': False, 'message': f'Detection error: {str(e)}'})

# --- Debug and Test Routes (Kept from original, may need adjustment for new face_utils) ---
@app.route('/debug_face_test')
def debug_face_test():
    if 'user_id' not in session or session.get('user_type') != 'student':
        return jsonify({'error': 'Please login as student first'})

    student_obj = Student.query.get(session['user_id'])
    if not student_obj: return jsonify({'error': 'Student not found'})
    if not student_obj.get_face_encoding().any(): return jsonify({'error': 'No face encoding for student'})

    stored_encoding = student_obj.get_face_encoding()
    # compare_faces returns boolean, compare_faces_advanced returns (bool, similarity)
    is_match, similarity = face_utils.compare_faces_advanced(stored_encoding, stored_encoding, tolerance=0.6)

    return jsonify({
        'student': student_obj.name,
        'roll_number': student_obj.roll_number,
        'encoding_shape': stored_encoding.shape if stored_encoding is not None else "N/A",
        'self_match_status': is_match,
        'self_match_similarity': similarity,
        'tolerance': 0.6,
        'config_tolerance': app.config.get('FACE_RECOGNITION_TOLERANCE'),
        'debug_mode': app.config.get('DEBUG_MODE', False),
        'bypass_face_recognition': app.config.get('BYPASS_FACE_RECOGNITION', False),
        'disable_anti_spoofing': app.config.get('DISABLE_ANTI_SPOOFING', False)
    })

@app.route('/toggle_production_mode') # This name is a bit misleading, it toggles debug features
def toggle_production_mode():
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'error': 'Only teachers can toggle debug modes'})

    app.config['BYPASS_FACE_RECOGNITION'] = not app.config.get('BYPASS_FACE_RECOGNITION', False)
    app.config['DISABLE_ANTI_SPOOFING'] = not app.config.get('DISABLE_ANTI_SPOOFING', False)
    # Ensure ULTRA_RELAXED_MODE is also toggled if DISABLE_ANTI_SPOOFING is
    if app.config['DISABLE_ANTI_SPOOFING']:
        app.config['ULTRA_RELAXED_MODE'] = True
    else: # If enabling anti-spoofing, turn off ultra-relaxed unless specifically set
        app.config['ULTRA_RELAXED_MODE'] = app.config.get('RELAXED_ANTI_SPOOFING', False)


    return jsonify({
        'message': 'Debug modes toggled',
        'bypass_face_recognition': app.config.get('BYPASS_FACE_RECOGNITION'),
        'disable_anti_spoofing': app.config.get('DISABLE_ANTI_SPOOFING'),
        'ultra_relaxed_mode': app.config.get('ULTRA_RELAXED_MODE'),
        'mode': 'DEBUG' if app.config.get('BYPASS_FACE_RECOGNITION') else 'STANDARD'
    })

@app.route('/emergency_bypass')
def emergency_bypass():
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'error': 'Only teachers can use emergency bypass'})

    app.config['BYPASS_FACE_RECOGNITION'] = True
    app.config['DISABLE_ANTI_SPOOFING'] = True
    app.config['ULTRA_RELAXED_MODE'] = True # Part of disabling anti-spoofing effectively
    # app.config['DISABLE_PRIVACY_CHECK'] = True # This was in original, but no privacy check implemented
    print("🚨 EMERGENCY BYPASS ACTIVATED 🚨")
    return jsonify({
        'message': 'Emergency bypass activated - Face Rec & Anti-Spoofing Bypassed/Disabled',
        'bypass_face_recognition': app.config['BYPASS_FACE_RECOGNITION'],
        'disable_anti_spoofing': app.config['DISABLE_ANTI_SPOOFING'],
        'ultra_relaxed_mode': app.config['ULTRA_RELAXED_MODE'],
        'mode': 'EMERGENCY_BYPASS'
    })

@app.route('/debug_system_status')
def debug_system_status():
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'error': 'Only teachers can access debug status'})
    return jsonify({
        'system_status': 'operational',
        'face_recognition_ready': True, # Basic check
        'config': {key: val for key, val in app.config.items() if key.isupper()},
        'database_tables': {
            'students': Student.query.count(),
            'teachers': Teacher.query.count(),
            'attendance': Attendance.query.count()
        }
    })

@app.route('/fix_camera_permissions') # Kept as is
def fix_camera_permissions():
    return """
    <html>...
    </html>
    """


if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Ensure tables are created on run
    app.run(debug=app.config.get('DEBUG_MODE', True), host='0.0.0.0', port=5000)
