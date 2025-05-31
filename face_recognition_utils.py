import cv2
try:
    import numpy as np
except ImportError:
    # Fallback for systems without numpy
    np = None
import base64
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class FaceRecognitionUtils:
    """Simplified and robust face recognition utilities"""
    
    def __init__(self):
        """Initialize face recognition utilities"""
        try:
            # Load face detector
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            # Detection parameters
            self.scale_factor = 1.1
            self.min_neighbors = 5
            self.min_face_size = (80, 80)
            self.tolerance = 0.5  # Face matching tolerance
            
            logger.info("Face recognition utilities initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing face recognition: {e}")
            raise

    def base64_to_image(self, base64_string):
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
            
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            return None

    def detect_faces(self, image):
        """Detect faces in an image"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Enhance image quality
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                enhanced,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def extract_face_encoding(self, image_input):
        """Extract face encoding from image"""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                image = cv2.imread(image_input)
                if image is None:
                    logger.error(f"Could not load image from {image_input}")
                    return None
            else:
                # NumPy array
                image = image_input
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                logger.warning("No faces detected in image")
                return None
            
            if len(faces) > 1:
                logger.warning(f"Multiple faces detected ({len(faces)}), using the largest one")
            
            # Get the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract face region with some padding
            padding = 20
            y1 = max(0, y - padding)
            y2 = min(image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(image.shape[1], x + w + padding)
            
            face_region = image[y1:y2, x1:x2]
            
            # Resize to standard size
            face_resized = cv2.resize(face_region, (160, 160))
            
            # Convert to grayscale for encoding
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Create a simple feature vector
            # This is a simplified approach - in production, you might want to use
            # more sophisticated methods like FaceNet or similar
            face_encoding = self._create_face_encoding(face_gray)
            
            logger.info("Face encoding extracted successfully")
            return face_encoding
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {e}")
            return None

    def _create_face_encoding(self, face_image):
        """Create a simple face encoding using image features"""
        try:
            # Normalize the image
            normalized = cv2.equalizeHist(face_image)
            
            # Calculate LBP (Local Binary Pattern) features
            lbp = self._calculate_lbp(normalized)
            
            # Calculate histogram features
            hist = cv2.calcHist([normalized], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-7)  # Normalize
            
            # Combine features
            encoding = np.concatenate([
                lbp.flatten()[:500],  # First 500 LBP features
                hist[:256]            # Histogram features
            ])
            
            # Normalize final encoding
            encoding = encoding / (np.linalg.norm(encoding) + 1e-7)
            
            return encoding
            
        except Exception as e:
            logger.error(f"Error creating face encoding: {e}")
            return None

    def _calculate_lbp(self, image, radius=3, n_points=24):
        """Calculate Local Binary Pattern features"""
        try:
            height, width = image.shape
            lbp_image = np.zeros((height, width), dtype=np.uint8)
            
            for i in range(radius, height - radius):
                for j in range(radius, width - radius):
                    center = image[i, j]
                    binary_string = ''
                    
                    # Sample points around the center
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        
                        if 0 <= x < height and 0 <= y < width:
                            binary_string += '1' if image[x, y] >= center else '0'
                        else:
                            binary_string += '0'
                    
                    lbp_image[i, j] = int(binary_string, 2) % 256
            
            return lbp_image
            
        except Exception as e:
            logger.error(f"Error calculating LBP: {e}")
            return np.zeros_like(image)

    def compare_faces(self, encoding1, encoding2):
        """Compare two face encodings"""
        try:
            if encoding1 is None or encoding2 is None:
                return {'is_match': False, 'confidence': 0.0}
            
            # Ensure encodings are numpy arrays
            if not isinstance(encoding1, np.ndarray):
                encoding1 = np.array(encoding1)
            if not isinstance(encoding2, np.ndarray):
                encoding2 = np.array(encoding2)
            
            # Ensure same length
            min_len = min(len(encoding1), len(encoding2))
            encoding1 = encoding1[:min_len]
            encoding2 = encoding2[:min_len]
            
            # Calculate similarity using cosine similarity
            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
            
            # Convert to distance (lower is better match)
            distance = 1 - similarity
            
            # Determine if match
            is_match = distance < self.tolerance
            confidence = max(0, 1 - distance)  # Convert to confidence score
            
            logger.info(f"Face comparison - Distance: {distance:.3f}, Match: {is_match}, Confidence: {confidence:.3f}")
            
            return {
                'is_match': is_match,
                'confidence': confidence,
                'distance': distance
            }
            
        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return {'is_match': False, 'confidence': 0.0}

    def save_image(self, image, file_path):
        """Save image to file"""
        try:
            cv2.imwrite(file_path, image)
            logger.info(f"Image saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False

    def validate_image_quality(self, image):
        """Validate image quality for face recognition"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Check brightness
            brightness = np.mean(gray)
            
            # Check contrast
            contrast = np.std(gray)
            
            # Check if image is too dark or too bright
            if brightness < 40:
                return {'valid': False, 'reason': 'Image is too dark'}
            elif brightness > 220:
                return {'valid': False, 'reason': 'Image is too bright'}
            elif contrast < 15:
                return {'valid': False, 'reason': 'Image has low contrast'}
            
            # Check image size
            height, width = gray.shape
            if height < 100 or width < 100:
                return {'valid': False, 'reason': 'Image resolution is too low'}
            
            return {'valid': True, 'brightness': brightness, 'contrast': contrast}
            
        except Exception as e:
            logger.error(f"Error validating image quality: {e}")
            return {'valid': False, 'reason': 'Error analyzing image'}
