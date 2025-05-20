from flask import render_template, redirect, url_for, request, jsonify
from app import app
import numpy as np
import io
from PIL import Image
from utils.face_recognition import detect_faces_from_webcam

@app.route('/')
def index():
    """Homepage route"""
    return render_template('index.html')

@app.route('/detect-face', methods=['POST'])
def detect_face():
    """API endpoint to detect faces in uploaded image"""
    if 'frame' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image from request
        image_file = request.files['frame']
        image_data = image_file.read()
        
        # Use our custom face detection function
        face_locations = detect_faces_from_webcam(image_data)
        
        # Return face locations
        return jsonify({
            'success': True,
            'faces': face_locations
        })
    except Exception as e:
        app.logger.error(f"Error in face detection: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/mark-notification-read/<int:notification_id>', methods=['POST'])
def mark_notification_read(notification_id):
    """Mark a notification as read"""
    from models import Notification
    from app import db
    from flask_login import current_user, login_required
    
    @login_required
    def process_notification():
        if not current_user.is_authenticated:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        notification = Notification.query.filter_by(
            id=notification_id, 
            user_id=current_user.id
        ).first()
        
        if not notification:
            return jsonify({'success': False, 'message': 'Notification not found'}), 404
        
        notification.read = True
        db.session.commit()
        
        return jsonify({'success': True})
    
    return process_notification()

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500
