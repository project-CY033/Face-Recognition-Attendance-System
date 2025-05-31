import os
import csv
import io
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from werkzeug.utils import secure_filename
from PIL import Image
import base64

# Configure logging
logger = logging.getLogger(__name__)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_save_file(file, upload_folder: str, filename_prefix: str = "") -> Optional[str]:
    """Securely save uploaded file"""
    try:
        if file and allowed_file(file.filename):
            # Generate secure filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename)
            filename = f"{filename_prefix}_{timestamp}_{original_filename}"
            
            # Ensure upload folder exists
            os.makedirs(upload_folder, exist_ok=True)
            
            # Save file
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            
            return filename
        return None
        
    except Exception as e:
        logger.error(f"File save error: {e}")
        return None

def validate_image_file(file_path: str, max_size_mb: int = 10) -> Dict[str, Any]:
    """Validate image file"""
    try:
        if not os.path.exists(file_path):
            return {'valid': False, 'error': 'File does not exist'}
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > max_size_mb * 1024 * 1024:
            return {'valid': False, 'error': f'File too large (max {max_size_mb}MB)'}
        
        # Try to open as image
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                format = img.format
                
                # Check minimum dimensions
                if width < 100 or height < 100:
                    return {'valid': False, 'error': 'Image too small (minimum 100x100)'}
                
                # Check maximum dimensions
                if width > 2048 or height > 2048:
                    return {'valid': False, 'error': 'Image too large (maximum 2048x2048)'}
                
                return {
                    'valid': True,
                    'width': width,
                    'height': height,
                    'format': format,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                }
                
        except Exception as e:
            return {'valid': False, 'error': 'Invalid image file'}
            
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return {'valid': False, 'error': 'Validation failed'}

def compress_image(image_path: str, quality: int = 85, max_width: int = 800) -> bool:
    """Compress image to reduce file size"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize if too large
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Save with compression
            img.save(image_path, 'JPEG', quality=quality, optimize=True)
            
        return True
        
    except Exception as e:
        logger.error(f"Image compression error: {e}")
        return False

def create_thumbnail(image_path: str, thumbnail_size: tuple = (150, 150)) -> Optional[str]:
    """Create thumbnail of image"""
    try:
        thumbnail_path = image_path.rsplit('.', 1)[0] + '_thumb.jpg'
        
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Create thumbnail
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            img.save(thumbnail_path, 'JPEG', quality=80)
            
        return thumbnail_path
        
    except Exception as e:
        logger.error(f"Thumbnail creation error: {e}")
        return None

def process_attendance_export(attendance_records: List, format: str = 'csv') -> io.StringIO:
    """Process attendance records for export"""
    try:
        output = io.StringIO()
        
        if format.lower() == 'csv':
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'Roll Number', 'Student Name', 'Date', 'Time In', 
                'Status', 'Confidence Score'
            ])
            
            # Write data
            for record in attendance_records:
                writer.writerow([
                    record.roll_number,
                    record.student_name,
                    record.date.strftime('%Y-%m-%d'),
                    record.time_in.strftime('%H:%M:%S'),
                    record.status,
                    f"{record.confidence_score:.2f}" if record.confidence_score else "N/A"
                ])
        
        output.seek(0)
        return output
        
    except Exception as e:
        logger.error(f"Export processing error: {e}")
        return io.StringIO()

def calculate_attendance_statistics(attendance_records: List) -> Dict[str, Any]:
    """Calculate comprehensive attendance statistics"""
    try:
        if not attendance_records:
            return {
                'total_records': 0,
                'present_count': 0,
                'absent_count': 0,
                'attendance_rate': 0.0,
                'daily_stats': {},
                'weekly_stats': {},
                'monthly_stats': {}
            }
        
        total_records = len(attendance_records)
        present_count = sum(1 for r in attendance_records if r.status == 'Present')
        absent_count = total_records - present_count
        attendance_rate = (present_count / total_records * 100) if total_records > 0 else 0
        
        # Daily statistics
        daily_stats = {}
        for record in attendance_records:
            date_str = record.date.strftime('%Y-%m-%d')
            if date_str not in daily_stats:
                daily_stats[date_str] = {'present': 0, 'total': 0}
            daily_stats[date_str]['total'] += 1
            if record.status == 'Present':
                daily_stats[date_str]['present'] += 1
        
        # Add percentage to daily stats
        for date_str in daily_stats:
            stats = daily_stats[date_str]
            stats['percentage'] = (stats['present'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        return {
            'total_records': total_records,
            'present_count': present_count,
            'absent_count': absent_count,
            'attendance_rate': round(attendance_rate, 2),
            'daily_stats': daily_stats,
            'date_range': {
                'start': min(r.date for r in attendance_records).strftime('%Y-%m-%d'),
                'end': max(r.date for r in attendance_records).strftime('%Y-%m-%d')
            }
        }
        
    except Exception as e:
        logger.error(f"Statistics calculation error: {e}")
        return {
            'total_records': 0,
            'present_count': 0,
            'absent_count': 0,
            'attendance_rate': 0.0,
            'error': str(e)
        }

def format_time_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    try:
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
            
    except Exception:
        return "N/A"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    try:
        # Remove or replace dangerous characters
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        sanitized = filename
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Limit length
        if len(sanitized) > 100:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:95] + ext
        
        return sanitized
        
    except Exception as e:
        logger.error(f"Filename sanitization error: {e}")
        return "default_file"

def validate_form_data(form_data: Dict, required_fields: List[str]) -> Dict[str, Any]:
    """Validate form data"""
    try:
        errors = []
        warnings = []
        
        # Check required fields
        for field in required_fields:
            if field not in form_data or not form_data[field].strip():
                errors.append(f"{field.replace('_', ' ').title()} is required")
        
        # Email validation (basic)
        if 'email' in form_data:
            email = form_data['email'].strip()
            if email and '@' not in email:
                errors.append("Invalid email format")
        
        # Phone validation (basic)
        if 'phone' in form_data:
            phone = form_data['phone'].strip()
            if phone and not phone.replace('+', '').replace('-', '').replace(' ', '').isdigit():
                warnings.append("Phone number format may be invalid")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
        
    except Exception as e:
        logger.error(f"Form validation error: {e}")
        return {
            'valid': False,
            'errors': ['Validation failed'],
            'warnings': []
        }

def generate_report_filename(report_type: str, filters: Dict = None) -> str:
    """Generate filename for reports"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{report_type}_{timestamp}"
        
        if filters:
            if 'date_from' in filters and filters['date_from']:
                filename += f"_from_{filters['date_from']}"
            if 'date_to' in filters and filters['date_to']:
                filename += f"_to_{filters['date_to']}"
            if 'roll_number' in filters and filters['roll_number']:
                filename += f"_roll_{filters['roll_number']}"
        
        return sanitize_filename(filename)
        
    except Exception as e:
        logger.error(f"Report filename generation error: {e}")
        return f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def log_user_activity(user_id: Optional[int], action: str, details: str = "", ip_address: str = "", user_agent: str = ""):
    """Log user activity for security and auditing"""
    try:
        # This would typically write to a log file or database
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'user_id': user_id,
            'action': action,
            'details': details,
            'ip_address': ip_address,
            'user_agent': user_agent[:500]  # Limit length
        }
        
        logger.info(f"User Activity: {log_entry}")
        
    except Exception as e:
        logger.error(f"Activity logging error: {e}")
