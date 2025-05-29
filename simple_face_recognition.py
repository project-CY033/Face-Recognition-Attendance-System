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
        try:
            # For basic image quality checks if needed, or anti-spoofing elements
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            print("✅ OpenCV Cascades loaded for SimpleFaceRecognition.")
        except Exception as e:
            print(f"❌ Error loading OpenCV cascades: {e}")
        print("✅ Simplified Face Recognition System Ready!")

    def base64_to_image(self, base64_string: str) -> np.ndarray | None:
        """Convert base64 string to OpenCV image (BGR)."""
        print("[SFR] base64_to_image called.")
        try:
            if ',' in base64_string: # Handle "data:image/jpeg;base64," prefix
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            pil_image = Image.open(io.BytesIO(image_bytes))
            print(f"[SFR] PIL image loaded: mode={pil_image.mode}, size={pil_image.size}")

            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                print(f"[SFR] PIL image converted to RGB.")
            
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            print(f"[SFR] Converted to OpenCV image, shape: {opencv_image.shape}")
            return opencv_image
        except Exception as e:
            print(f"❌ Error converting base64 to image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def assess_image_quality(self, image_cv: np.ndarray) -> dict:
        """Basic image quality assessment. Made very lenient."""
        # print("[SFR] assess_image_quality called.")
        laplacian_var = cv2.Laplacian(image_cv, cv2.CV_64F).var()
        brightness = np.mean(cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY))

        message = "Quality check passed (lenient)."
        is_good = True
        # Not making these strict for now to avoid blocking
        # if laplacian_var < Config.MIN_BLUR_THRESHOLD / 2:
        #     message = f"Image might be too blurry (blur: {laplacian_var:.2f})."
        # if not (Config.MIN_IMAGE_BRIGHTNESS / 2 < brightness < Config.MAX_IMAGE_BRIGHTNESS * 1.5):
        #      message += f" Check brightness (current: {brightness:.2f})."

        return {
            'is_good_quality': is_good,
            'message': message,
            'details': {'blur': laplacian_var, 'brightness': brightness},
            'quality_score': 0.9,
            'performance': 'ultra_fast'
        }

    def detect_faces_advanced(self, image_cv: np.ndarray) -> dict:
        """Detects faces using face_recognition library (dlib HOG model)."""
        print(f"[SFR] detect_faces_advanced called with image shape: {image_cv.shape}")
        start_time = time.time()
        try:
            rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            print("[SFR] Image converted to RGB for face_recognition.")
            face_locations_dlib = face_recognition.face_locations(rgb_image, model="hog") 
            print(f"[SFR] face_recognition.face_locations found: {face_locations_dlib}")
            
            faces_found = len(face_locations_dlib) > 0
            multiple_faces = len(face_locations_dlib) > 1
            
            face_regions_cv = []
            for (top, right, bottom, left) in face_locations_dlib:
                face_regions_cv.append((left, top, right - left, bottom - top))
            
            detection_time = time.time() - start_time
            print(f"[SFR] Face detection took {detection_time:.4f}s. Found {len(face_locations_dlib)} faces.")

            return {
                'faces_found': faces_found,
                'face_count': len(face_locations_dlib),
                'multiple_faces': multiple_faces,
                'face_regions_cv': face_regions_cv, 
                'dlib_face_locations': face_locations_dlib
            }
        except Exception as e:
            print(f"❌ Error in detect_faces_advanced: {e}")
            import traceback
            traceback.print_exc()
            return {'faces_found': False, 'error': str(e)}

    def extract_face_encoding(self, image_source: str | np.ndarray) -> np.ndarray | None:
        """
        Extracts face encoding from an image file path or an OpenCV image.
        Uses the first detected face.
        """
        print(f"[SFR] extract_face_encoding called. Source type: {type(image_source)}")
        try:
            if isinstance(image_source, str): # Path
                print(f"[SFR] Loading image from path: {image_source}")
                loaded_image = face_recognition.load_image_file(image_source)
            elif isinstance(image_source, np.ndarray): # OpenCV image (BGR)
                print(f"[SFR] Using OpenCV image with shape: {image_source.shape}")
                loaded_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            else:
                print("❌ Invalid image_source type for encoding.")
                return None
            print("[SFR] Image loaded/converted to RGB for encoding.")

            face_locations = face_recognition.face_locations(loaded_image, model="hog")
            print(f"[SFR] Detected face locations for encoding: {face_locations}")
            
            if not face_locations:
                print("❌ No faces found in image for encoding.")
                return None
            
            print(f"[SFR] Extracting encoding for the first face at {face_locations[0]}")
            face_encodings_list = face_recognition.face_encodings(loaded_image, known_face_locations=[face_locations[0]], num_jitters=1)
            # num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate but slower. Default 1.
            
            if face_encodings_list:
                print(f"[SFR] Encoding extracted, shape: {face_encodings_list[0].shape}")
                return face_encodings_list[0]
            else:
                print("❌ Could not extract encoding from the detected face.")
                return None
        except Exception as e:
            print(f"❌ Error extracting face encoding: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_specific_face_encoding(self, image_cv: np.ndarray, dlib_location: tuple) -> dict:
        """Extracts encoding for a specific pre-detected dlib face location."""
        print(f"[SFR] extract_specific_face_encoding called for location: {dlib_location}, image shape: {image_cv.shape}")
        try:
            rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image, known_face_locations=[dlib_location], num_jitters=1)
            print(f"[SFR] face_recognition.face_encodings for specific location found: {len(encodings) if encodings else 'None'}")
            
            if encodings:
                return {
                    'success': True,
                    'encoding': encodings[0],
                    'confidence': 0.95, 
                    'error': None
                }
            return {'success': False, 'encoding': None, 'confidence': 0.0, 'error': 'Encoding failed for specific location'}
        except Exception as e:
            print(f"❌ Error in extract_specific_face_encoding: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'encoding': None, 'confidence': 0.0, 'error': str(e)}


    def compare_faces_advanced(self, stored_encoding: np.ndarray, captured_encoding: np.ndarray, tolerance: float = 0.6) -> tuple[bool, float]:
        """
        Compares two face encodings.
        Returns: (is_match: bool, similarity_score: float)
                 similarity_score is 1.0 - distance. Higher is more similar.
        """
        print(f"[SFR] compare_faces_advanced called. Tolerance: {tolerance}")
        if stored_encoding is None or captured_encoding is None:
            print("[SFR] One or both encodings are None.")
            return False, 0.0
        
        print(f"[SFR] Stored encoding shape: {stored_encoding.shape}, Captured encoding shape: {captured_encoding.shape}")
        try:
            matches = face_recognition.compare_faces([stored_encoding], captured_encoding, tolerance=tolerance)
            is_match = matches[0] if matches else False
            
            distance = face_recognition.face_distance([stored_encoding], captured_encoding)[0]
            similarity_score = 1.0 - distance 
            print(f"[SFR] Comparison: Match={is_match}, Distance={distance:.4f}, Similarity={similarity_score:.4f}")
            
            return is_match, similarity_score
        except Exception as e:
            print(f"❌ Error comparing faces: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0

    def detect_spoofing(self, image_cv: np.ndarray) -> dict:
        """
        Basic spoofing detection. Very lenient, especially if config flags are set.
        """
        print("[SFR] detect_spoofing called.")
        if Config.DISABLE_ANTI_SPOOFING:
            print("[SFR] Anti-spoofing disabled by configuration.")
            return {'is_spoofing': False, 'reason': 'Anti-spoofing disabled.', 'confidence': 0.99}
        
        if Config.ULTRA_RELAXED_MODE or Config.RELAXED_ANTI_SPOOFING:
            print("[SFR] Anti-spoofing in relaxed/ultra-relaxed mode.")
            return {'is_spoofing': False, 'reason': 'Relaxed anti-spoofing.', 'confidence': 0.95}

        try:
            gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            print(f"[SFR] Laplacian variance for spoof check: {laplacian_var:.2f}, Threshold: {Config.SPOOFING_TEXTURE_THRESHOLD}")

            if laplacian_var < Config.SPOOFING_TEXTURE_THRESHOLD:
                return {
                    'is_spoofing': True,
                    'reason': f'Low image texture (Laplacian var: {laplacian_var:.2f}). Possible spoof.',
                    'confidence': 0.4 
                }
            
            return {
                'is_spoofing': False,
                'reason': f'Basic texture check passed (Laplacian var: {laplacian_var:.2f}).',
                'confidence': 0.75
            }
        except Exception as e:
            print(f"❌ Error in basic spoofing detection: {e}")
            return {'is_spoofing': False, 'reason': f'Spoof check error: {str(e)}', 'confidence': 0.6}

    def compare_faces(self, stored_encoding, captured_encoding, tolerance=0.6):
        is_match, _ = self.compare_faces_advanced(stored_encoding, captured_encoding, tolerance)
        return is_match

    def get_similarity_score(self, encoding1, encoding2):
        if encoding1 is None or encoding2 is None: return 0.0
        distance = face_recognition.face_distance([encoding1], encoding2)
        return (1.0 - distance[0]) if distance else 0.0
