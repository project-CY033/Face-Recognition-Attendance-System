import cv2
import numpy as np
import base64
from PIL import Image
import io
import time
import face_recognition # Main library for simplified operations
from config import Config # For anti-spoofing config access

class SimpleFaceRecognition:
    def __init__(self):
        """Initialize the simplified face recognition system"""
        print("🚀 Initializing Simplified Face Recognition System...")
        # For basic image quality checks if needed, or anti-spoofing elements
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        print("✅ Simplified Face Recognition System Ready!")

    def base64_to_image(self, base64_string: str) -> np.ndarray | None:
        """Convert base64 string to OpenCV image (BGR)."""
        try:
            if ',' in base64_string: # Handle "data:image/jpeg;base64," prefix
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            pil_image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if it's not, then to BGR for OpenCV
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            print(f"❌ Error converting base64 to image: {e}")
            return None

    def assess_image_quality(self, image_cv: np.ndarray) -> dict:
        """Basic image quality assessment. Made very lenient."""
        # For now, assume quality is good to focus on core recognition
        # Placeholder for more advanced checks if needed later
        laplacian_var = cv2.Laplacian(image_cv, cv2.CV_64F).var()
        brightness = np.mean(cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY))

        message = "Quality check passed (lenient)."
        is_good = True

        if laplacian_var < Config.MIN_BLUR_THRESHOLD / 2: # Very blurry
            message = f"Image might be too blurry (blur: {laplacian_var:.2f})."
            # is_good = False # Keep it lenient
        if not (Config.MIN_IMAGE_BRIGHTNESS / 2 < brightness < Config.MAX_IMAGE_BRIGHTNESS * 1.5):
             message += f" Check brightness (current: {brightness:.2f})."
            # is_good = False # Keep it lenient

        return {
            'is_good_quality': is_good,
            'message': message,
            'details': {'blur': laplacian_var, 'brightness': brightness},
            'quality_score': 0.9 if is_good else 0.5, # Arbitrary
            'performance': 'ultra_fast'
        }

    def detect_faces_advanced(self, image_cv: np.ndarray) -> dict:
        """Detects faces using face_recognition library (dlib HOG model)."""
        start_time = time.time()
        try:
            rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            # model="hog" is faster, model="cnn" is more accurate but much slower
            face_locations_dlib = face_recognition.face_locations(rgb_image, model="hog") 
            
            faces_found = len(face_locations_dlib) > 0
            multiple_faces = len(face_locations_dlib) > 1
            
            # Convert dlib rects (top, right, bottom, left) to OpenCV (x, y, w, h)
            face_regions_cv = []
            for (top, right, bottom, left) in face_locations_dlib:
                face_regions_cv.append((left, top, right - left, bottom - top))
            
            detection_time = time.time() - start_time
            # print(f"Face detection took {detection_time:.4f}s")

            return {
                'faces_found': faces_found,
                'face_count': len(face_locations_dlib),
                'multiple_faces': multiple_faces,
                'face_regions_cv': face_regions_cv, 
                'dlib_face_locations': face_locations_dlib # Pass this to encoding function
            }
        except Exception as e:
            print(f"❌ Error in detect_faces_advanced: {e}")
            return {'faces_found': False, 'error': str(e)}

    def extract_face_encoding(self, image_source: str | np.ndarray) -> np.ndarray | None:
        """
        Extracts face encoding from an image file path or an OpenCV image.
        Uses the first detected face.
        """
        try:
            if isinstance(image_source, str): # Path
                loaded_image = face_recognition.load_image_file(image_source) # Loads as RGB
            elif isinstance(image_source, np.ndarray): # OpenCV image (BGR)
                loaded_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            else:
                print("❌ Invalid image_source type for encoding.")
                return None

            # Detect face locations (using HOG model for speed)
            # Could also use "cnn" for more accuracy but slower
            face_locations = face_recognition.face_locations(loaded_image, model="hog")
            
            if not face_locations:
                print("❌ No faces found in image for encoding.")
                return None
            
            # Extract encoding for the first detected face
            # face_recognition.face_encodings returns a list of encodings
            face_encodings_list = face_recognition.face_encodings(loaded_image, known_face_locations=[face_locations[0]])
            
            if face_encodings_list:
                return face_encodings_list[0] # Return the 128-d encoding array
            else:
                print("❌ Could not extract encoding from the detected face.")
                return None
        except Exception as e:
            print(f"❌ Error extracting face encoding: {e}")
            return None

    def extract_specific_face_encoding(self, image_cv: np.ndarray, dlib_location: tuple) -> dict:
        """Extracts encoding for a specific pre-detected dlib face location."""
        try:
            rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            # dlib_location is expected as (top, right, bottom, left)
            encodings = face_recognition.face_encodings(rgb_image, known_face_locations=[dlib_location])
            
            if encodings:
                return {
                    'success': True,
                    'encoding': encodings[0],
                    'confidence': 0.95, # Static confidence for successful encoding
                    'error': None
                }
            return {'success': False, 'encoding': None, 'confidence': 0.0, 'error': 'Encoding failed for specific location'}
        except Exception as e:
            print(f"❌ Error in extract_specific_face_encoding: {e}")
            return {'success': False, 'error': str(e)}

    def compare_faces_advanced(self, stored_encoding: np.ndarray, captured_encoding: np.ndarray, tolerance: float = 0.6) -> tuple[bool, float]:
        """
        Compares two face encodings.
        Returns: (is_match: bool, similarity_score: float)
                 similarity_score is 1.0 - distance. Higher is more similar.
        """
        if stored_encoding is None or captured_encoding is None:
            return False, 0.0
        
        try:
            # compare_faces returns a list of booleans
            matches = face_recognition.compare_faces([stored_encoding], captured_encoding, tolerance=tolerance)
            is_match = matches[0] if matches else False
            
            # face_distance returns Euclidean distance; lower is better
            distance = face_recognition.face_distance([stored_encoding], captured_encoding)[0]
            similarity_score = 1.0 - distance # Convert distance to a similarity score (0 to 1 range approx)
            
            return is_match, similarity_score
        except Exception as e:
            print(f"❌ Error comparing faces: {e}")
            return False, 0.0

    def detect_spoofing(self, image_cv: np.ndarray) -> dict:
        """
        Basic spoofing detection. Very lenient, especially if config flags are set.
        This is NOT a robust anti-spoofing solution.
        """
        # Check config flags for bypassing or relaxing anti-spoofing
        if Config.DISABLE_ANTI_SPOOFING:
            return {
                'is_spoofing': False,
                'reason': 'Anti-spoofing disabled by configuration.',
                'confidence': 0.99 # High confidence it's live because check is off
            }
        if Config.ULTRA_RELAXED_MODE or Config.RELAXED_ANTI_SPOOFING:
             return {
                'is_spoofing': False,
                'reason': 'Anti-spoofing check passed due to relaxed/ultra-relaxed mode.',
                'confidence': 0.95 # High confidence it's live
            }

        # If no bypass, perform a very basic texture check
        try:
            gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

            # Config.SPOOFING_TEXTURE_THRESHOLD: higher value means more texture needed to pass
            if laplacian_var < Config.SPOOFING_TEXTURE_THRESHOLD:
                return {
                    'is_spoofing': True,
                    'reason': f'Low image texture (Laplacian variance: {laplacian_var:.2f}). Possible spoof.',
                    'confidence': 0.4 # Lower confidence it's live
                }
            
            return {
                'is_spoofing': False,
                'reason': f'Basic texture check passed (Laplacian variance: {laplacian_var:.2f}).',
                'confidence': 0.75 # Medium confidence it's live
            }
        except Exception as e:
            print(f"❌ Error in basic spoofing detection: {e}")
            # In case of error, assume it's not a spoof to avoid blocking valid users
            return {
                'is_spoofing': False,
                'reason': f'Spoofing check error, assuming live: {str(e)}',
                'confidence': 0.6 # Neutral confidence
            }

    # --- Alias methods for compatibility with existing calls if needed ---
    def compare_faces(self, stored_encoding, captured_encoding, tolerance=0.6):
        """Returns True if faces match, False otherwise."""
        is_match, _ = self.compare_faces_advanced(stored_encoding, captured_encoding, tolerance)
        return is_match

    def get_similarity_score(self, encoding1, encoding2):
        """Calculates a similarity score between two encodings."""
        if encoding1 is None or encoding2 is None:
            return 0.0
        # face_distance returns a list of distances, we take the first one.
        distance = face_recognition.face_distance([encoding1], encoding2)
        if distance:
            return 1.0 - distance[0] # Higher is more similar
        return 0.0
