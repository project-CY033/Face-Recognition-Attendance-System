# Simple Face Recognition Attendance System

A clean and simple web-based attendance system using face recognition technology built with Flask and OpenCV.

## Features

- **Student Registration**: Register students with live camera photo capture
- **Face Recognition**: Simple and effective face recognition for attendance
- **Live Camera**: Real-time camera integration
- **Dashboard**: Clean dashboards for students and teachers
- **Database Management**: Complete CRUD operations
- **Responsive Design**: Mobile-friendly interface

## Technologies Used

- **Backend**: Flask, SQLAlchemy
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **Computer Vision**: OpenCV, NumPy
- **Database**: SQLite
- **Camera**: WebRTC API for live camera access


# Screenshot of Project

<details>
  <summary>Click to see Screenshots</summary>

---
---
# Home page
![image](https://github.com/user-attachments/assets/67b58828-191c-4f85-87d4-b8341cd2968f)

---

# Student Registration
![image](https://github.com/user-attachments/assets/65f9f26b-0c2e-4767-b76f-73547e696c27)

---
![image](https://github.com/user-attachments/assets/9567963f-912b-4a4a-ace8-f4e44148ad02)


---
# Students dashboard

![image](https://github.com/user-attachments/assets/7cbfce11-cefd-4162-8f6c-697960f58b18)

# Attendance Records
![image](https://github.com/user-attachments/assets/a1351fc8-a175-46a2-90a5-d4d1990d763c)



# Teacher Dashboard
![image](https://github.com/user-attachments/assets/1437f8e2-7540-4614-9929-ec55a88cc953)

# Teacher Attendance Records
![image](https://github.com/user-attachments/assets/5e0d15f0-9f89-4873-870f-6b2d4fdbd59d)
![image](https://github.com/user-attachments/assets/ee1cadf1-d816-4c5f-9465-6800520a80d3)



  
</details>



---
---


## Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

```
pip install scipy

pip install cmake
pip install dlib

```






# activate virtual environment
```
.\venv\Scripts\activate

```

# Go to the project directory
```
cd Face-Recognition-Attendance-System   

```

# Install official CMake for Windows:
- ### Go to: https://cmake.org/download/

# Download the Windows x64 Installer (e.g., cmake-3.xx.x-windows-x86_64.msi)

- ## During installation:

✅ Check "Add CMake to the system PATH for all users"

Complete the install

🧪 3. Verify it's working:
Open a new terminal (very important) and run:











3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

### Student Registration
1. Go to the home page
2. Click "Register as Student"
3. Fill in your details
4. Use the camera to capture your photo
5. Submit the form

### Marking Attendance
1. Login with your roll number
2. Go to "Mark Attendance"
3. Allow camera access
4. Capture your photo
5. System will verify your face and mark attendance

### Teacher Dashboard
1. Register as a teacher
2. Login with teacher credentials
3. Access student management
4. View attendance records

## Project Structure

```
face-attendance-system/
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── models.py                   # Database models
├── simple_face_recognition.py  # Face recognition utilities
├── requirements.txt            # Python dependencies
├── templates/                  # HTML templates
├── static/                     # Static files
├── frontend/                   # Complete frontend package
└── instance/                   # Database files
```

## Clean and Simple

This is a cleaned version with:
- ✅ Simple face recognition using OpenCV
- ✅ Essential features only
- ✅ Clean code structure
- ✅ Easy to understand and modify
- ✅ Production-ready frontend package included

## Frontend Package

The `frontend/` folder contains a complete frontend package that can be shared with other developers:
- Complete HTML templates
- API documentation
- Sample configuration
- Requirements and setup guide

## License

This project is licensed under the MIT License.
