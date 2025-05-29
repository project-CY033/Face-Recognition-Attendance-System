from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime, date
from sqlalchemy import inspect as sql_inspect # Renamed to avoid conflict
import cv2 # For saving image in student_register
import traceback # For detailed error logging

from config import Config
from models import db, Student, Teacher, Attendance
from simple_face_recognition import SimpleFaceRecognition

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
            db.create_all()
            inspector = sql_inspect(db.engine)
            if inspector.has_table(Student.__tablename__):
                columns = [col['name'] for col in inspector.get_columns(Student.__tablename__)]
                if 'semester' not in columns:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("!!! WARNING: 'semester' column is missing from 'student' table.      !!!")
                    # ... (rest of the warning message) ...
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        except Exception as e:
            print(f"⚠️ Database initialization/check error: {e}")
            traceback.print_exc()
            try:
                db.create_all()
                print("✅ Database tables ensured/re-attempted.")
            except Exception as e2:
                print(f"⚠️ Persistent database error: {e2}")
                traceback.print_exc()
        create_tables.called = True


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/student_register', methods=['GET', 'POST'])
def student_register():
    """Student registration page"""
    if request.method == 'POST':
        print("[App] Student registration POST request received.")
        try:
            roll_number = request.form['roll_number']
            name = request.form['name']
            email = request.form['email']
            phone = request.form['phone']
            course = request.form['course']
            year = request.form['year']
            semester = request.form['semester']
            subject = request.form['subject']
            print(f"[App] Form data: Roll={roll_number}, Name={name}")

            existing_student = Student.query.filter_by(roll_number=roll_number).first()
            if existing_student:
                flash('Student with this roll number already exists!', 'error')
                return render_template('student_register.html')

            existing_email = Student.query.filter_by(email=email).first()
            if existing_email:
                flash('Student with this email already exists!', 'error')
                return render_template('student_register.html')

            photo_data = request.form.get('photo_data') 
            if not photo_data:
                flash('Please capture a photo using the camera.', 'error')
                print("[App] No photo data provided for registration.")
                return render_template('student_register.html')

            print("📷 Processing camera captured photo from base64 for registration...")
            filename = None 
            photo_path = None 
            try:
                image_cv = face_utils.base64_to_image(photo_data)
                if image_cv is None:
                    flash('Invalid camera photo data. Please try again.', 'error')
                    print("[App] Registration: base64_to_image returned None.")
                    return render_template('student_register.html')
                print(f"[App] Registration: Image decoded from base64, shape: {image_cv.shape}")

                filename = secure_filename(f"student_{roll_number}_camera.jpg")
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(photo_path, image_cv) # Save the OpenCV image
                print(f"📸 Registration: Photo saved to {photo_path}")

                face_encoding = face_utils.extract_face_encoding(image_cv) # Pass the cv image

            except Exception as e:
                print(f"Error processing camera photo in student_register: {e}")
                traceback.print_exc()
                flash(f'Error processing camera photo: {str(e)}', 'error')
                return render_template('student_register.html')

            if face_encoding is None:
                flash('No face detected or could not extract features. Please ensure your face is clearly visible and try again.', 'error')
                print("[App] Registration: face_encoding is None after extraction attempt.")
                if filename and os.path.exists(photo_path):
                    try:
                        os.remove(photo_path)
                        print(f"🗑️ Removed photo {photo_path} due to encoding failure.")
                    except OSError as e_os:
                        print(f"Error removing photo {photo_path}: {e_os}")
                return render_template('student_register.html')
            
            print(f"✅ Registration: Face encoding extracted successfully. Shape: {face_encoding.shape}")

            student = Student(
                roll_number=roll_number, name=name, email=email, phone=phone,
                course=course, year=year, semester=semester, subject=subject,
                photo_path=filename
            )
            student.set_face_encoding(face_encoding)

            db.session.add(student)
            db.session.commit()
            print(f"[App] Student {roll_number} registered and committed to DB.")
            flash('Student registered successfully with face recognition setup!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            print(f"Error during student registration: {e}")
            traceback.print_exc()
            flash(f'Error during registration: {str(e)}', 'error')
            db.session.rollback()
    return render_template('student_register.html')

@app.route('/teacher_register', methods=['GET', 'POST'])
def teacher_register():
    """Teacher registration page"""
    if request.method == 'POST':
        print("[App] Teacher registration POST request received.")
        try:
            teacher_id_form = request.form['teacher_id']
            name = request.form['name']
            email = request.form['email']
            phone = request.form['phone']
            department = request.form['department']
            subject = request.form['subject']
            print(f"[App] Teacher form data: ID={teacher_id_form}, Name={name}")

            existing_teacher = Teacher.query.filter_by(teacher_id=teacher_id_form).first()
            if existing_teacher:
                flash('Teacher with this ID already exists!', 'error')
                return render_template('teacher_register.html')

            existing_email = Teacher.query.filter_by(email=email).first()
            if existing_email:
                flash('Teacher with this email already exists!', 'error')
                return render_template('teacher_register.html')

            photo_file = request.files['photo']
            if photo_file and photo_file.filename:
                filename = secure_filename(f"teacher_{teacher_id_form}_{photo_file.filename}")
                photo_path_full = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                photo_file.save(photo_path_full)
                print(f"📸 Teacher photo saved to {photo_path_full}")

                face_encoding = face_utils.extract_face_encoding(photo_path_full) # Pass file path
                if face_encoding is None:
                    flash('No face detected in the uploaded photo. Please upload a clear photo.', 'error')
                    print("[App] Teacher reg: No face encoding from uploaded photo.")
                    try:
                        os.remove(photo_path_full)
                    except OSError as e_os:
                        print(f"Error removing teacher photo {photo_path_full}: {e_os}")
                    return render_template('teacher_register.html')
                print(f"✅ Teacher reg: Face encoding extracted, shape: {face_encoding.shape}")

                teacher = Teacher(
                    teacher_id=teacher_id_form, name=name, email=email, phone=phone,
                    department=department, subject=subject, photo_path=filename
                )
                teacher.set_face_encoding(face_encoding)
                db.session.add(teacher)
                db.session.commit()
                print(f"[App] Teacher {teacher_id_form} registered and committed to DB.")
                flash('Teacher registered successfully!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Please upload a photo for face recognition.', 'error')
        except Exception as e:
            print(f"Error during teacher registration: {e}")
            traceback.print_exc()
            flash(f'Error during registration: {str(e)}', 'error')
            db.session.rollback()
    return render_template('teacher_register.html')


@app.route('/login', methods=['POST'])
def login():
    user_id_form = request.form['user_id']
    user_type = request.form['user_type']
    print(f"[App] Login attempt: UserID={user_id_form}, Type={user_type}")

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
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/student_dashboard')
def student_dashboard():
    if 'user_id' not in session or session.get('user_type') != 'student':
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('index'))
    student_obj = Student.query.get(session['user_id'])
    if not student_obj:
        flash('Student not found.', 'error')
        session.clear()
        return redirect(url_for('index'))

    total_attendance = Attendance.query.filter_by(student_id=student_obj.id).count()
    present_days = Attendance.query.filter_by(student_id=student_obj.id, status='Present').count()
    absent_days = total_attendance - present_days 
    percentage = round((present_days / total_attendance * 100), 1) if total_attendance > 0 else 0
    attendance_stats = {'total_days': total_attendance, 'present_days': present_days, 'absent_days': absent_days, 'percentage': percentage}
    recent_attendance = Attendance.query.filter_by(student_id=student_obj.id).order_by(Attendance.created_at.desc()).limit(5).all()
    return render_template('student_dashboard.html', student=student_obj, attendance_stats=attendance_stats, recent_attendance=recent_attendance)

@app.route('/teacher_dashboard')
def teacher_dashboard():
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('index'))
    teacher_obj = Teacher.query.get(session['user_id'])
    if not teacher_obj:
        flash('Teacher account not found.', 'error')
        session.clear()
        return redirect(url_for('index'))

    total_students = Student.query.count()
    today_date = date.today()
    today_present = Attendance.query.filter_by(date=today_date, status='Present').count()
    today_absent = total_students - today_present if total_students >= today_present else 0
    attendance_percentage = round((today_present / total_students * 100), 1) if total_students > 0 else 0
    return render_template('teacher_dashboard.html', teacher=teacher_obj, total_students=total_students, today_present=today_present, today_absent=today_absent, attendance_percentage=attendance_percentage)

@app.route('/mark_attendance')
def mark_attendance():
    if 'user_id' not in session or session.get('user_type') != 'student':
        flash('Please login to mark attendance.', 'error')
        return redirect(url_for('index'))
    return render_template('mark_attendance.html')

@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    print("[App] process_attendance called.")
    try:
        if 'user_id' not in session or session.get('user_type') != 'student':
            return jsonify({'success': False, 'message': 'Authentication required.', 'error_code': 'AUTH_REQUIRED'})

        student_obj = Student.query.get(session['user_id'])
        if not student_obj:
            return jsonify({'success': False, 'message': 'Student account not found.', 'error_code': 'STUDENT_NOT_FOUND'})
        print(f"🎯 Processing attendance for: {student_obj.name} (Roll: {student_obj.roll_number})")

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data received.', 'error_code': 'NO_IMAGE_DATA'})

        print("📷 Converting base64 image for attendance...")
        try:
            image_cv = face_utils.base64_to_image(data['image'])
            if image_cv is None:
                print("[App] Attendance: base64_to_image returned None.")
                return jsonify({'success': False, 'message': 'Invalid image format.', 'error_code': 'INVALID_IMAGE'})
            print(f"[App] Attendance: Image decoded, shape: {image_cv.shape}")
            if image_cv.shape[0] < 100 or image_cv.shape[1] < 100:
                return jsonify({'success': False, 'message': 'Image too small.', 'error_code': 'IMAGE_TOO_SMALL'})
        except Exception as e:
            print(f"Image processing error in process_attendance: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Image processing error: {str(e)}.', 'error_code': 'IMAGE_PROCESSING_ERROR'})

        quality_result = face_utils.assess_image_quality(image_cv)
        print(f"[App] Image quality: {quality_result}")
        # Not blocking on quality for now, to simplify debugging recognition
        # if not quality_result['is_good_quality']:
        #     return jsonify({'success': False, 'message': f'Image quality issue: {quality_result["message"]}.', 'error_code': 'POOR_IMAGE_QUALITY'})

        print("🔍 Detecting faces in attendance image...")
        face_detection_result = face_utils.detect_faces_advanced(image_cv)
        print(f"[App] Face detection result: {face_detection_result}")
        
        if not face_detection_result.get('faces_found'):
            return jsonify({'success': False, 'message': '😕 No face detected.', 'error_code': 'NO_FACE_DETECTED'})
        if face_detection_result.get('multiple_faces'):
            return jsonify({'success': False, 'message': '👥 Multiple faces detected.', 'error_code': 'MULTIPLE_FACES'})

        print("🛡️ Performing anti-spoofing checks...")
        spoofing_result = face_utils.detect_spoofing(image_cv)
        print(f"[App] Spoofing check result: {spoofing_result}")
        if spoofing_result['is_spoofing']:
            # Only block if NOT in relaxed/disabled mode
            if not (Config.DISABLE_ANTI_SPOOFING or Config.ULTRA_RELAXED_MODE or Config.RELAXED_ANTI_SPOOFING) :
                 return jsonify({
                    'success': False, 
                    'message': f'🚫 Security alert: {spoofing_result["reason"]}. Live camera only.',
                    'error_code': 'SPOOFING_DETECTED',
                    'security_details': spoofing_result
                })
            else:
                print("[App] Spoofing detected but overridden by relaxed/disabled config.")


        print("🧠 Extracting face encoding for attendance...")
        if not face_detection_result.get('dlib_face_locations') or not face_detection_result['dlib_face_locations']:
            print("❌ No dlib_face_locations found in detection_result for attendance.")
            return jsonify({'success': False, 'message': 'Internal error: Face location data missing.', 'error_code': 'INTERNAL_DETECTION_ERROR'})
        
        dlib_location = face_detection_result['dlib_face_locations'][0]
        encoding_result = face_utils.extract_specific_face_encoding(image_cv, dlib_location)
        print(f"[App] Encoding result for attendance: {encoding_result}")

        if not encoding_result or not encoding_result.get('success'):
            error_msg = encoding_result.get('error', 'Unknown') if encoding_result else "None"
            return jsonify({'success': False, 'message': f'Face analysis failed: {error_msg}', 'error_code': 'ENCODING_FAILED'})

        captured_encoding = encoding_result['encoding']
        # print(f"✅ Attendance: Face encoding extracted. Confidence: {encoding_result.get('confidence', 0.0):.3f}")

        print(f"📋 Retrieving stored face data for {student_obj.roll_number}...")
        stored_encoding = student_obj.get_face_encoding()
        if stored_encoding is None:
            print(f"[App] No stored encoding for student {student_obj.roll_number}")
            return jsonify({'success': False, 'message': 'No registered face data. Please re-register.', 'error_code': 'NO_STORED_ENCODING'})
        # print(f"[App] Stored encoding retrieved, shape: {stored_encoding.shape}")

        print("⚡ Performing face matching for attendance...")
        tolerance = app.config.get('FACE_RECOGNITION_TOLERANCE', 0.6)
        primary_match, similarity_score = face_utils.compare_faces_advanced(stored_encoding, captured_encoding, tolerance=tolerance)
        # print(f"📊 Attendance Match: Status={primary_match}, Similarity={similarity_score:.3f}, Tolerance={tolerance}")

        if primary_match:
            today_date = date.today()
            existing_attendance = Attendance.query.filter_by(student_id=student_obj.id, date=today_date).first()
            if existing_attendance:
                return jsonify({'success': False, 'message': f'✅ Already marked for today at {existing_attendance.time_in.strftime("%I:%M %p")}', 'error_code': 'ALREADY_MARKED'})

            current_time = datetime.now()
            attendance_record = Attendance(
                student_id=student_obj.id, roll_number=student_obj.roll_number, student_name=student_obj.name,
                date=today_date, time_in=current_time.time(), status='Present'
            )
            db.session.add(attendance_record)
            db.session.commit()
            print(f"🎉 Attendance marked successfully for {student_obj.name}")
            
            total_days = Attendance.query.filter_by(student_id=student_obj.id).count()
            present_days = Attendance.query.filter_by(student_id=student_obj.id, status='Present').count()
            attendance_percentage = round((present_days / total_days * 100), 1) if total_days > 0 else 100

            return jsonify({
                'success': True, 'message': f'🎉 Attendance marked for {student_obj.name}!',
                'details': {
                    'student_name': student_obj.name, 'roll_number': student_obj.roll_number,
                    'time_marked': current_time.strftime("%I:%M %p"), 'date': today_date.strftime("%B %d, %Y"),
                    'similarity_score': round(similarity_score, 3),
                    'confidence': round(encoding_result.get('confidence', 0.0), 3), 
                    'total_attendance_days': total_days, 'attendance_percentage': attendance_percentage
                }
            })
        else:
            print(f"❌ Face recognition failed for {student_obj.name}")
            return jsonify({
                'success': False, 'message': '❌ Face does not match registered photo. Please try again.',
                'error_code': 'FACE_MISMATCH',
                'details': {
                    'similarity_score': round(similarity_score, 3),
                    'required_similarity_threshold_for_match_at_tolerance': round(1 - tolerance, 3),
                    'suggestions': ["Ensure good lighting", "Look directly at camera"]
                }
            })
    except Exception as e:
        print(f"💥 Outer Error in process_attendance: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'System error occurred. Please contact support.', 'error_code': 'SYSTEM_ERROR', 'details': str(e) if app.debug else "Internal Server Error"})


@app.route('/view_attendance')
def view_attendance():
    if 'user_id' not in session:
        flash('Please login to view attendance.', 'error')
        return redirect(url_for('index'))

    query = Attendance.query
    user_type = session.get('user_type')
    roll_number_search = request.args.get('roll_number', '')
    date_from_str = request.args.get('date_from', '')
    date_to_str = request.args.get('date_to', '')

    if user_type == 'student':
        student_obj = Student.query.get(session['user_id'])
        if not student_obj:
            flash('Student not found.', 'error'); return redirect(url_for('index'))
        query = query.filter_by(student_id=student_obj.id)
    elif user_type == 'teacher':
        if roll_number_search:
            query = query.filter(Attendance.roll_number.contains(roll_number_search))
    else:
        flash('Invalid user type.', 'error'); return redirect(url_for('index'))
    
    try:
        if date_from_str: query = query.filter(Attendance.date >= datetime.strptime(date_from_str, '%Y-%m-%d').date())
        if date_to_str: query = query.filter(Attendance.date <= datetime.strptime(date_to_str, '%Y-%m-%d').date())
    except ValueError:
        flash('Invalid date format. Please use YYYY-MM-DD.', 'error')

    attendance_records = query.order_by(Attendance.date.desc(), Attendance.time_in.desc()).all()
    return render_template('view_attendance.html', attendance_records=attendance_records)


@app.route('/database_viewer')
def database_viewer():
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को Database Access की अनुमति है!', 'error')
        return redirect(url_for('index'))
    teacher_obj = Teacher.query.get(session['user_id'])
    if not teacher_obj:
        flash('Teacher account not found.', 'error'); return redirect(url_for('logout'))

    students_list = Student.query.all()
    teachers_list = Teacher.query.all()
    attendance_records_list = Attendance.query.order_by(Attendance.created_at.desc()).limit(100).all()
    stats = {
        'total_students': Student.query.count(), 'total_teachers': Teacher.query.count(),
        'total_attendance': Attendance.query.count(),
        'face_encodings': Student.query.filter(Student.face_encoding.isnot(None)).count() + \
                         Teacher.query.filter(Teacher.face_encoding.isnot(None)).count()
    }
    return render_template('database_viewer.html', students=students_list, teachers=teachers_list,
                         attendance_records=attendance_records_list, stats=stats, current_teacher=teacher_obj)

@app.route('/edit_attendance/<int:attendance_id>', methods=['GET', 'POST'])
def edit_attendance(attendance_id):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को Attendance Edit करने की अनुमति है!', 'error')
        return redirect(url_for('index'))
    attendance_record = Attendance.query.get_or_404(attendance_id)
    if request.method == 'POST':
        try:
            attendance_record.student_name = request.form['student_name']
            attendance_record.date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
            attendance_record.time_in = datetime.strptime(request.form['time_in'], '%H:%M').time()
            attendance_record.status = request.form['status']
            db.session.commit()
            flash(f'Attendance record updated for {attendance_record.student_name}!', 'success')
            return redirect(url_for('database_viewer'))
        except Exception as e:
            flash(f'Error updating attendance: {str(e)}', 'error'); db.session.rollback()
    return render_template('edit_attendance.html', attendance=attendance_record)

@app.route('/delete_attendance/<int:attendance_id>', methods=['POST'])
def delete_attendance(attendance_id):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})
    try:
        attendance_record = Attendance.query.get_or_404(attendance_id)
        name = attendance_record.student_name
        db.session.delete(attendance_record)
        db.session.commit()
        return jsonify({'success': True, 'message': f'Attendance for {name} deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/bulk_attendance_action', methods=['POST'])
def bulk_attendance_action():
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})
    try:
        data = request.get_json()
        action = data.get('action')
        ids = data.get('attendance_ids', [])
        if not ids: return jsonify({'success': False, 'message': 'No records selected'})
        count = 0
        message = ""
        if action == 'delete':
            Attendance.query.filter(Attendance.id.in_(ids)).delete(synchronize_session=False)
            count = len(ids)
            message = f'{count} attendance records deleted.'
        elif action == 'mark_absent':
            records_to_update = Attendance.query.filter(Attendance.id.in_(ids)).all()
            for record in records_to_update:
                record.status = 'Absent'
                count +=1
            message = f'{count} records marked as Absent.'
        else:
            return jsonify({'success': False, 'message': 'Invalid action'})
        
        if count > 0: db.session.commit()
        return jsonify({'success': True, 'message': message})
    except Exception as e:
        db.session.rollback(); return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/edit_student/<int:student_id_param>', methods=['GET', 'POST'])
def edit_student(student_id_param):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को अनुमति है!', 'error'); return redirect(url_for('index'))
    student_obj = Student.query.get_or_404(student_id_param)
    if request.method == 'POST':
        try:
            student_obj.name = request.form.get('name', student_obj.name)
            student_obj.email = request.form.get('email', student_obj.email)
            student_obj.course = request.form.get('course', student_obj.course)
            student_obj.year = request.form.get('year', student_obj.year)
            student_obj.semester = request.form.get('semester', student_obj.semester)
            student_obj.subject = request.form.get('subject', student_obj.subject)
            db.session.commit()
            flash('Student information updated!', 'success'); return redirect(url_for('database_viewer'))
        except Exception as e:
            db.session.rollback(); flash(f'Error updating student: {str(e)}', 'error')
    return render_template('edit_student.html', student=student_obj)

@app.route('/delete_student/<int:student_id_param>', methods=['POST'])
def delete_student(student_id_param):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})
    try:
        student_obj = Student.query.get_or_404(student_id_param)
        name = student_obj.name
        Attendance.query.filter_by(student_id=student_obj.id).delete()
        if student_obj.photo_path:
            try: os.remove(os.path.join(app.config['UPLOAD_FOLDER'], student_obj.photo_path))
            except OSError as e: print(f"Error deleting student photo {student_obj.photo_path}: {e}")
        db.session.delete(student_obj)
        db.session.commit()
        return jsonify({'success': True, 'message': f'Student {name} deleted!'})
    except Exception as e:
        db.session.rollback(); return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/edit_teacher/<int:teacher_id_param>', methods=['GET', 'POST'])
def edit_teacher(teacher_id_param):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        flash('केवल Teachers को अनुमति है!', 'error'); return redirect(url_for('index'))
    teacher_obj = Teacher.query.get_or_404(teacher_id_param)
    if request.method == 'POST':
        try:
            teacher_obj.name = request.form.get('name', teacher_obj.name)
            teacher_obj.email = request.form.get('email', teacher_obj.email)
            teacher_obj.department = request.form.get('department', teacher_obj.department)
            teacher_obj.subject = request.form.get('subject', teacher_obj.subject)
            db.session.commit()
            flash('Teacher information updated!', 'success'); return redirect(url_for('database_viewer'))
        except Exception as e:
            db.session.rollback(); flash(f'Error updating teacher: {str(e)}', 'error')
    return render_template('edit_teacher.html', teacher=teacher_obj)

@app.route('/delete_teacher/<int:teacher_id_param>', methods=['POST'])
def delete_teacher(teacher_id_param):
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'success': False, 'message': 'केवल Teachers को अनुमति है!'})
    try:
        teacher_obj = Teacher.query.get_or_404(teacher_id_param)
        name = teacher_obj.name
        if teacher_obj.photo_path:
            try: os.remove(os.path.join(app.config['UPLOAD_FOLDER'], teacher_obj.photo_path))
            except OSError as e: print(f"Error deleting teacher photo {teacher_obj.photo_path}: {e}")
        db.session.delete(teacher_obj)
        db.session.commit()
        return jsonify({'success': True, 'message': f'Teacher {name} deleted!'})
    except Exception as e:
        db.session.rollback(); return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/detect_face_realtime', methods=['POST'])
def detect_face_realtime():
    print("[App] /detect_face_realtime called")
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'face_detected': False, 'message': 'No image data'})

        image_cv = face_utils.base64_to_image(data['image'])
        if image_cv is None:
            return jsonify({'face_detected': False, 'message': 'Invalid image format'})

        detection_result = face_utils.detect_faces_advanced(image_cv)
        # print(f"[App] Realtime detection result: {detection_result}")
        
        if detection_result.get('faces_found'):
            if detection_result.get('multiple_faces'):
                return jsonify({'face_detected': False, 'message': f'Multiple faces ({detection_result["face_count"]}).', 'face_count': detection_result["face_count"]})
            
            face_coords = None
            if detection_result.get('dlib_face_locations') and detection_result['dlib_face_locations']:
                top, right, bottom, left = detection_result['dlib_face_locations'][0]
                img_h, img_w = image_cv.shape[:2]
                face_coords = {
                    'x': (left / img_w) * 100, 'y': (top / img_h) * 100,
                    'width': ((right - left) / img_w) * 100, 'height': ((bottom - top) / img_h) * 100
                }
            return jsonify({'face_detected': True, 'message': 'Face detected!', 'face_coords': face_coords, 'quality': 'good'})
        else:
            return jsonify({'face_detected': False, 'message': 'No face detected.'})
    except Exception as e:
        print(f"Error in /detect_face_realtime: {e}"); traceback.print_exc()
        return jsonify({'face_detected': False, 'message': f'Detection error: {str(e)}'})

@app.route('/debug_face_test')
def debug_face_test():
    if 'user_id' not in session or session.get('user_type') != 'student':
        return jsonify({'error': 'Please login as student first'})
    student_obj = Student.query.get(session['user_id'])
    if not student_obj: return jsonify({'error': 'Student not found'})
    
    stored_encoding = student_obj.get_face_encoding()
    if stored_encoding is None or not stored_encoding.any(): # Check if None or empty
         return jsonify({'error': f'No face encoding found for student {student_obj.name}'})

    is_match, similarity = face_utils.compare_faces_advanced(stored_encoding, stored_encoding, tolerance=0.6)
    return jsonify({
        'student': student_obj.name, 'roll_number': student_obj.roll_number,
        'encoding_shape': stored_encoding.shape,
        'self_match_status': is_match, 'self_match_similarity': similarity, 'tolerance': 0.6,
        'config_tolerance': app.config.get('FACE_RECOGNITION_TOLERANCE'),
        'debug_mode': app.config.get('DEBUG_MODE'),
        'bypass_face_recognition': app.config.get('BYPASS_FACE_RECOGNITION'),
        'disable_anti_spoofing': app.config.get('DISABLE_ANTI_SPOOFING')
    })

@app.route('/toggle_production_mode')
def toggle_production_mode():
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'error': 'Only teachers can toggle debug modes'})
    app.config['BYPASS_FACE_RECOGNITION'] = not app.config.get('BYPASS_FACE_RECOGNITION', False)
    app.config['DISABLE_ANTI_SPOOFING'] = not app.config.get('DISABLE_ANTI_SPOOFING', False)
    if app.config['DISABLE_ANTI_SPOOFING']: app.config['ULTRA_RELAXED_MODE'] = True
    else: app.config['ULTRA_RELAXED_MODE'] = app.config.get('RELAXED_ANTI_SPOOFING', False)
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
    app.config['ULTRA_RELAXED_MODE'] = True
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
    
    config_subset = {key: val for key, val in app.config.items() if isinstance(val, (str, int, float, bool, list, dict, type(None))) and key.isupper()}

    return jsonify({
        'system_status': 'operational', 'face_recognition_ready': True,
        'config': config_subset,
        'database_tables': {
            'students': Student.query.count(), 'teachers': Teacher.query.count(),
            'attendance': Attendance.query.count()
        }
    })

@app.route('/fix_camera_permissions')
def fix_camera_permissions():
    return """
    <html>
    <head><title>Fix Camera Issues</title></head>
    <body style="font-family: Arial; padding: 20px; max-width: 800px; margin: 0 auto;">
        <h1>🔧 Fix Camera Permission Issues</h1>
        <h2>Chrome/Edge:</h2><ol><li>Click the camera icon in the address bar</li><li>Select "Always allow" for camera access</li><li>Refresh the page</li></ol>
        <h2>Firefox:</h2><ol><li>Click the shield icon in the address bar</li><li>Click "Turn off Blocking for this site"</li><li>Refresh the page</li></ol>
        <h2>Safari:</h2><ol><li>Go to Safari > Preferences > Websites > Camera</li><li>Set this website to "Allow"</li><li>Refresh the page</li></ol>
        <h2>Still having issues?</h2><ul><li>Check if another application is using your camera</li><li>Try restarting your browser</li><li>Make sure your camera is properly connected</li><li>Try using a different browser</li></ul>
        <p><a href="/" style="background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">← Back to App</a></p>
    </body>
    </html>
    """

@app.route('/toggle_relaxed_mode') # This was defined twice, removed the second definition
def toggle_relaxed_mode():
    if 'user_id' not in session or session.get('user_type') != 'teacher':
        return jsonify({'error': 'Only teachers can toggle relaxed mode'})
    app.config['RELAXED_ANTI_SPOOFING'] = not app.config.get('RELAXED_ANTI_SPOOFING', False)
    # If relaxed anti-spoofing is turned on, ultra_relaxed_mode should also be on for consistency if anti-spoofing isn't disabled
    if app.config['RELAXED_ANTI_SPOOFING'] and not app.config['DISABLE_ANTI_SPOOFING']:
        app.config['ULTRA_RELAXED_MODE'] = True
    elif not app.config['DISABLE_ANTI_SPOOFING']: # If turning relaxed off, and anti-spoofing is enabled, turn ultra_relaxed off
        app.config['ULTRA_RELAXED_MODE'] = False

    return jsonify({
        'message': f'Relaxed anti-spoofing mode {"enabled" if app.config.get("RELAXED_ANTI_SPOOFING") else "disabled"}',
        'relaxed_anti_spoofing': app.config.get('RELAXED_ANTI_SPOOFING'),
        'ultra_relaxed_mode': app.config.get('ULTRA_RELAXED_MODE'),
        'mode': 'RELAXED_DEVELOPMENT' if app.config.get('RELAXED_ANTI_SPOOFING') else 'STANDARD'
    })

if __name__ == '__main__':
    with app.app_context():
        # The create_tables() function handles this via @app.before_request
        # db.create_all() 
        pass
    app.run(debug=app.config.get('DEBUG_MODE', True), host='0.0.0.0', port=5000)
