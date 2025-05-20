import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import current_app

def send_email(recipient_list, subject, body, sender_name=None):
    """
    Send email to recipients
    
    Args:
        recipient_list (list): List of recipient email addresses
        subject (str): Email subject
        body (str): Email body
        sender_name (str, optional): Name to display as sender
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Get email configuration from app config
        mail_server = current_app.config.get('MAIL_SERVER')
        mail_port = current_app.config.get('MAIL_PORT')
        mail_use_tls = current_app.config.get('MAIL_USE_TLS')
        mail_username = current_app.config.get('MAIL_USERNAME')
        mail_password = current_app.config.get('MAIL_PASSWORD')
        mail_default_sender = current_app.config.get('MAIL_DEFAULT_SENDER')
        
        # Check if email is configured
        if not all([mail_server, mail_port, mail_username, mail_password, mail_default_sender]):
            print("Email not configured properly")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{sender_name} <{mail_default_sender}>" if sender_name else mail_default_sender
        msg['To'] = ", ".join(recipient_list)
        msg['Subject'] = subject
        
        # Attach body
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to server and send
        server = smtplib.SMTP(mail_server, mail_port)
        if mail_use_tls:
            server.starttls()
        server.login(mail_username, mail_password)
        server.sendmail(mail_default_sender, recipient_list, msg.as_string())
        server.quit()
        
        return True
    
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def send_attendance_notification(student_email, student_name, subject_name, date_str, marked_by):
    """
    Send attendance notification email
    
    Args:
        student_email (str): Student's email address
        student_name (str): Student's name
        subject_name (str): Subject name
        date_str (str): Date string
        marked_by (str): How attendance was marked
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    subject = f"Attendance Marked - {subject_name}"
    
    method = "face recognition" if marked_by == "face" else "manual entry" if marked_by == "manual" else "your teacher"
    
    body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #4CAF50; color: white; padding: 10px; text-align: center; }}
            .content {{ padding: 20px; background-color: #f9f9f9; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Attendance Confirmation</h2>
            </div>
            <div class="content">
                <p>Hello {student_name},</p>
                <p>Your attendance has been marked for <strong>{subject_name}</strong> on <strong>{date_str}</strong> via {method}.</p>
                <p>If you believe this is an error, please contact your teacher or the system administrator.</p>
            </div>
            <div class="footer">
                <p>This is an automated email. Please do not reply.</p>
                <p>© Attendance System</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email([student_email], subject, body, "Attendance System")
