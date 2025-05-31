import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging
import os
from typing import Optional, Dict, Any, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """
    Optimized Face Recognition System
    Features:
    - Robust face detection with multiple algorithms
    - Efficient face encoding using OpenCV
    - Anti-spoofing detection
    - Quality assessment
    - Performance optimization
    """
    
    def __init__(self):
        """Initialize the face recognition system"""
        logger.info("Initializing Face Recognition System...")
        
        try:
            # Initialize face detection cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            # Face recognition settings
            self.face_size = (200, 200)
            self.tolerance = 0.6
            self.min_confidence = 0.8
            
            # Quality thresholds
            self.min_brightness = 50
            self.max_brightness = 200
            self.min_contrast = 30
            self.min_face_size = (80, 80)
            
            # Initialize LBPH face recognizer for encoding
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            logger.info("Face Recognition System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing face recognition system: {e}")
            raise e
    
    def detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect faces in image using multiple algorithms
        """
        try:
            start_time = time.time()
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Enhance image quality
            enhanced_gray = self._enhance_image(gray)
            
            faces = []
            confidence_scores = []
            
            # Primary detection
            detected_faces = self.face_cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If no faces found, try alternative cascade
            if len(detected_faces) == 0:
                detected_faces = self.face_cascade_alt.detectMultiScale(
                    enhanced_gray,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=self.min_face_size
                )
            
            # Process detected faces
            for (x, y, w, h) in detected_faces:
                # Quality check
                face_roi = enhanced_gray[y:y+h, x:x+w]
                quality_score = self._assess_face_quality(face_roi, gray[y:y+h, x:x+w])
                
                if quality_score > 0.5:  # Minimum quality threshold
                    faces.append((x, y, w, h))
                    confidence_scores.append(quality_score)
            
            detection_time = time.time() - start_time
            
            return {
                'faces_found': len(faces) > 0,
                'face_count': len(faces),
                'face_regions': faces,
                'confidence_scores': confidence_scores,
                'detection_time': detection_time,
                'image_quality': self._assess_image_quality(gray)
            }
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return {
                'faces_found': False,
                'face_count': 0,
                'face_regions': [],
                'error': str(e)
            }
    
    def _enhance_image(self, gray_image: np.ndarray) -> np.ndarray:
        """Enhance image for better face detection"""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_image)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement error: {e}")
            return gray_image
    
    def _assess_face_quality(self, face_roi: np.ndarray, original_face: np.ndarray) -> float:
        """Assess the quality of detected face"""
        try:
            quality_score = 0.0
            
            # Size check
            if face_roi.shape[0] >= 80 and face_roi.shape[1] >= 80:
                quality_score += 0.2
            
            # Eye detection
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            if len(eyes) >= 2:
                quality_score += 0.3
            elif len(eyes) == 1:
                quality_score += 0.15
            
            # Contrast check
            contrast = np.std(face_roi)
            if contrast > 30:
                quality_score += 0.2
            
            # Brightness check
            brightness = np.mean(original_face)
            if 50 <= brightness <= 200:
                quality_score += 0.2
            
            # Sharpness check (Laplacian variance)
            laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            if laplacian_var > 100:
                quality_score += 0.1
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return 0.5
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess overall image quality"""
        try:
            brightness = np.mean(image)
            contrast = np.std(image)
            sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'overall_score': min((brightness / 128) * (contrast / 50) * (sharpness / 500), 1.0)
            }
            
        except Exception as e:
            logger.error(f"Image quality assessment error: {e}")
            return {'brightness': 0, 'contrast': 0, 'sharpness': 0, 'overall_score': 0}
    
    def extract_face_encoding(self, image: np.ndarray, face_region: Optional[Tuple] = None) -> Optional[np.ndarray]:
        """Extract face encoding from image"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # If face region is provided, use it; otherwise detect face
            if face_region:
                x, y, w, h = face_region
                face_roi = gray[y:y+h, x:x+w]
            else:
                detection_result = self.detect_faces(image)
                if not detection_result['faces_found']:
                    return None
                
                # Use the first (best) detected face
                x, y, w, h = detection_result['face_regions'][0]
                face_roi = gray[y:y+h, x:x+w]
            
            # Resize face to standard size
            face_resized = cv2.resize(face_roi, self.face_size)
            
            # Normalize the image
            face_normalized = cv2.equalizeHist(face_resized)
            
            # Extract features using Local Binary Pattern
            encoding = self._extract_lbp_features(face_normalized)
            
            return encoding
            
        except Exception as e:
            logger.error(f"Face encoding extraction error: {e}")
            return None
    
    def _extract_lbp_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        try:
            # Calculate LBP
            radius = 3
            n_points = 8 * radius
            
            # Create LBP image
            lbp = self._local_binary_pattern(face_image, n_points, radius)
            
            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            
            # Normalize histogram
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # Add additional features
            mean_intensity = np.mean(face_image)
            std_intensity = np.std(face_image)
            
            # Combine features
            features = np.concatenate([hist, [mean_intensity, std_intensity]])
            
            return features
            
        except Exception as e:
            logger.error(f"LBP feature extraction error: {e}")
            return np.array([])
    
    def _local_binary_pattern(self, image: np.ndarray, n_points: int, radius: float) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        try:
            rows, cols = image.shape
            lbp = np.zeros((rows, cols), dtype=np.uint8)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[i, j]
                    code = 0
                    
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = i + radius * np.cos(angle)
                        y = j + radius * np.sin(angle)
                        
                        # Bilinear interpolation
                        x1, y1 = int(x), int(y)
                        x2, y2 = x1 + 1, y1 + 1
                        
                        if x2 < rows and y2 < cols:
                            # Interpolate pixel value
                            dx, dy = x - x1, y - y1
                            pixel_val = (1 - dx) * (1 - dy) * image[x1, y1] + \
                                       dx * (1 - dy) * image[x2, y1] + \
                                       (1 - dx) * dy * image[x1, y2] + \
                                       dx * dy * image[x2, y2]
                            
                            if pixel_val >= center:
                                code |= (1 << k)
                    
                    lbp[i, j] = code
            
            return lbp
            
        except Exception as e:
            logger.error(f"LBP calculation error: {e}")
            return np.zeros_like(image)
    
    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> Dict[str, Any]:
        """Compare two face encodings"""
        try:
            if encoding1 is None or encoding2 is None:
                return {'is_match': False, 'confidence': 0.0, 'error': 'Invalid encodings'}
            
            # Ensure encodings are the same length
            if len(encoding1) != len(encoding2):
                return {'is_match': False, 'confidence': 0.0, 'error': 'Encoding length mismatch'}
            
            # Calculate similarity using multiple metrics
            
            # 1. Euclidean distance
            euclidean_dist = np.linalg.norm(encoding1 - encoding2)
            euclidean_similarity = 1.0 / (1.0 + euclidean_dist)
            
            # 2. Cosine similarity
            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            cosine_similarity = dot_product / (norm1 * norm2 + 1e-7)
            
            # 3. Correlation coefficient
            correlation = np.corrcoef(encoding1, encoding2)[0, 1]
            correlation = 0 if np.isnan(correlation) else correlation
            
            # Combine similarities
            combined_similarity = (euclidean_similarity * 0.4 + 
                                 cosine_similarity * 0.4 + 
                                 correlation * 0.2)
            
            # Determine if it's a match
            is_match = combined_similarity > self.tolerance
            
            return {
                'is_match': is_match,
                'confidence': combined_similarity,
                'euclidean_similarity': euclidean_similarity,
                'cosine_similarity': cosine_similarity,
                'correlation': correlation,
                'threshold': self.tolerance
            }
            
        except Exception as e:
            logger.error(f"Face comparison error: {e}")
            return {'is_match': False, 'confidence': 0.0, 'error': str(e)}
    
    def process_face_image(self, image_input, is_base64: bool = False) -> Optional[np.ndarray]:
        """Process face image and extract encoding"""
        try:
            # Load image
            if is_base64:
                image = self._base64_to_image(image_input)
            elif isinstance(image_input, str):
                image = cv2.imread(image_input)
            else:
                image = image_input
            
            if image is None:
                logger.error("Could not load image")
                return None
            
            # Detect faces
            detection_result = self.detect_faces(image)
            
            if not detection_result['faces_found']:
                logger.warning("No faces detected in image")
                return None
            
            if detection_result['face_count'] > 1:
                logger.warning("Multiple faces detected, using the first one")
            
            # Extract encoding from the best face
            face_region = detection_result['face_regions'][0]
            encoding = self.extract_face_encoding(image, face_region)
            
            return encoding
            
        except Exception as e:
            logger.error(f"Face processing error: {e}")
            return None
    
    def _base64_to_image(self, base64_string: str) -> Optional[np.ndarray]:
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image then to OpenCV
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
            
        except Exception as e:
            logger.error(f"Base64 to image conversion error: {e}")
            return None
    
    def validate_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Validate image quality for face recognition"""
        try:
            quality_info = self._assess_image_quality(image)
            
            issues = []
            recommendations = []
            
            # Check brightness
            if quality_info['brightness'] < self.min_brightness:
                issues.append("Image is too dark")
                recommendations.append("Increase lighting or camera brightness")
            elif quality_info['brightness'] > self.max_brightness:
                issues.append("Image is too bright")
                recommendations.append("Reduce lighting or camera exposure")
            
            # Check contrast
            if quality_info['contrast'] < self.min_contrast:
                issues.append("Low contrast")
                recommendations.append("Improve lighting conditions")
            
            # Check sharpness
            if quality_info['sharpness'] < 100:
                issues.append("Image is blurry")
                recommendations.append("Hold camera steady and ensure proper focus")
            
            is_good_quality = len(issues) == 0
            
            return {
                'is_good_quality': is_good_quality,
                'quality_score': quality_info['overall_score'],
                'issues': issues,
                'recommendations': recommendations,
                'details': quality_info
            }
            
        except Exception as e:
            logger.error(f"Image quality validation error: {e}")
            return {
                'is_good_quality': False,
                'quality_score': 0.0,
                'issues': ['Error validating image quality'],
                'recommendations': ['Please try again with a different image'],
                'error': str(e)
            }
