import os
import io
import cv2
import numpy as np
import pickle
import logging
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Initialize face detection
# We'll use a simplified approach with OpenCV
face_detector = None
try:
    # Try to get the cascade classifier path
    cascade_path = None
    haarcascades_dir = None
    
    # Check common locations for the haarcascades directory
    possible_locations = [
        # Look in the project directory first
        'haarcascades',
        os.path.join(os.getcwd(), 'haarcascades'),
        # Common system locations
        os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades'),
        '/usr/local/share/opencv4/haarcascades',
        '/usr/share/opencv4/haarcascades',
        '/usr/share/opencv/haarcascades'
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            haarcascades_dir = location
            break
    
    if haarcascades_dir:
        cascade_path = os.path.join(haarcascades_dir, 'haarcascade_frontalface_default.xml')
        
        if os.path.exists(cascade_path):
            face_detector = cv2.CascadeClassifier(cascade_path)
            logger.info(f"Loaded face detector from: {cascade_path}")
        else:
            logger.warning(f"Cascade file not found at: {cascade_path}")
    else:
        logger.warning("Could not find haarcascades directory")
        
except Exception as e:
    logger.error(f"Error initializing face detector: {str(e)}")

def encode_face(image_data):
    """
    Generate face encoding from image data using OpenCV
    
    Args:
        image_data (bytes): Image data in bytes format
    
    Returns:
        bytes: Pickled face features or None if no face detected
    """
    try:
        # If face detector is not available, save the image directly
        if face_detector is None:
            logger.warning("Face detector not available, using raw image instead")
            # Just resize and save the image as grayscale
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((100, 100)).convert('L')
            face_encoding = pickle.dumps(np.array(image))
            return face_encoding
        
        # Convert image data to numpy array
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Convert to RGB if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # If RGBA
            image_np = image_np[:, :, :3]
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            logger.warning("No face detected in the image")
            # Just resize and save the image as grayscale as a fallback
            image = image.resize((100, 100)).convert('L')
            face_encoding = pickle.dumps(np.array(image))
            return face_encoding
        
        # Extract the first face
        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        
        # Resize for consistency
        face_img = cv2.resize(face_img, (100, 100))
        
        # Store the face image as a pickle
        face_encoding = pickle.dumps(face_img)
        
        return face_encoding
    
    except Exception as e:
        logger.error(f"Error encoding face: {str(e)}")
        return None

def recognize_face(image_data, stored_encoding, threshold=70):
    """
    Compare face in image to stored encoding using OpenCV
    
    Args:
        image_data (bytes): Image data in bytes format
        stored_encoding (bytes): Pickled stored face features
        threshold (float): Confidence threshold (lower is stricter)
    
    Returns:
        bool: True if face matches, False otherwise
    """
    try:
        if stored_encoding is None:
            logger.warning("Stored encoding is None")
            return False
        
        # Convert image data to numpy array
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Unpickle stored face
        stored_face = pickle.loads(stored_encoding)
        
        # If face detector is not available, do simple comparison
        if face_detector is None:
            logger.warning("Face detector not available, using simple comparison")
            # Convert current image to grayscale and resize
            if len(image_np.shape) == 3:  # If color
                image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image_np
            
            image_gray = cv2.resize(image_gray, (100, 100))
            
            # Calculate MSE (Mean Squared Error) between images
            # Lower MSE means more similarity
            mse = np.mean((image_gray - stored_face) ** 2)
            max_mse = 255 * 255  # Maximum possible MSE for 8-bit images
            similarity = (1 - (mse / max_mse)) * 100  # Convert to percentage
            
            logger.debug(f"Face match similarity (MSE): {similarity}%")
            
            # Return True if similarity is above threshold
            return similarity > threshold
        
        # Convert to RGB if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # If RGBA
            image_np = image_np[:, :, :3]
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            logger.warning("No face detected in the image for recognition")
            return False  # No face detected
        
        # Extract the first face
        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        
        # Resize for consistency
        face_img = cv2.resize(face_img, (100, 100))
        
        # Compare faces using template matching
        result = cv2.matchTemplate(face_img, stored_face, cv2.TM_CCOEFF_NORMED)
        similarity = np.max(result) * 100  # Convert to percentage
        
        logger.debug(f"Face match similarity: {similarity}%")
        
        # Return True if similarity is above threshold
        return similarity > threshold
    
    except Exception as e:
        logger.error(f"Error recognizing face: {str(e)}")
        return False

def detect_faces_from_webcam(frame_data):
    """
    Detect faces in webcam frame using OpenCV
    
    Args:
        frame_data (bytes): Frame data in bytes format
    
    Returns:
        list: List of face locations [(top, right, bottom, left), ...]
    """
    try:
        # Convert frame data to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # If face detector is not available, return a generic face location
        # This is for demonstration purposes only
        if face_detector is None:
            logger.warning("Face detector not available, using generic face location")
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            face_size = min(width, height) // 4
            
            # Generate a face box at the center of the image
            top = center_y - face_size
            right = center_x + face_size
            bottom = center_y + face_size
            left = center_x - face_size
            
            return [(top, right, bottom, left)]
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        # Convert to (top, right, bottom, left) format
        face_locations = []
        for (x, y, w, h) in faces:
            top = y
            right = x + w
            bottom = y + h
            left = x
            face_locations.append((top, right, bottom, left))
        
        return face_locations
    
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return []
