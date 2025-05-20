/**
 * Attendance management JavaScript
 */

// Initialize attendance calendar view
function initAttendanceCalendar() {
    const calendarBody = document.getElementById('attendance-calendar-body');
    if (!calendarBody) return;
    
    // Add click handler for calendar cells
    calendarBody.addEventListener('click', (e) => {
        const cell = e.target.closest('td[data-date]');
        if (!cell) return;
        
        const date = cell.getAttribute('data-date');
        const studentId = cell.getAttribute('data-student');
        const subjectId = cell.getAttribute('data-subject');
        
        if (date && studentId && subjectId) {
            toggleAttendance(studentId, subjectId, date);
        }
    });
}

// Toggle attendance status for a student
async function toggleAttendance(studentId, subjectId, date) {
    try {
        // Get current status
        const cell = document.querySelector(`td[data-date="${date}"][data-student="${studentId}"][data-subject="${subjectId}"]`);
        const attendanceId = cell.getAttribute('data-attendance-id');
        
        // Show modal for confirmation and note
        const modal = new bootstrap.Modal(document.getElementById('attendanceModal'));
        const modalTitle = document.getElementById('attendanceModalLabel');
        const modalBody = document.getElementById('attendanceModalBody');
        const confirmBtn = document.getElementById('confirmAttendanceBtn');
        
        if (attendanceId) {
            // Student is present, confirm marking as absent
            modalTitle.textContent = 'Remove Attendance';
            modalBody.innerHTML = `
                <p>Are you sure you want to remove this attendance record?</p>
                <div class="mb-3">
                    <label for="attendanceNote" class="form-label">Note (optional):</label>
                    <textarea class="form-control" id="attendanceNote" rows="3" placeholder="Reason for removing attendance"></textarea>
                </div>
            `;
            
            confirmBtn.onclick = async () => {
                const note = document.getElementById('attendanceNote').value;
                await updateAttendanceRecord(attendanceId, 'delete', note);
                modal.hide();
            };
        } else {
            // Student is absent, confirm marking as present
            modalTitle.textContent = 'Mark Attendance';
            modalBody.innerHTML = `
                <p>Mark this student as present for ${date}?</p>
                <div class="mb-3">
                    <label for="attendanceNote" class="form-label">Note (optional):</label>
                    <textarea class="form-control" id="attendanceNote" rows="3" placeholder="Additional notes"></textarea>
                </div>
            `;
            
            confirmBtn.onclick = async () => {
                const note = document.getElementById('attendanceNote').value;
                await markAttendance(studentId, subjectId, date, note);
                modal.hide();
            };
        }
        
        modal.show();
    } catch (error) {
        console.error('Error toggling attendance:', error);
        showAlert('Error updating attendance. Please try again.', 'danger');
    }
}

// Update an existing attendance record
async function updateAttendanceRecord(attendanceId, action, note) {
    try {
        showLoading();
        
        const formData = new FormData();
        formData.append('attendance_id', attendanceId);
        formData.append('action', action);
        formData.append('note', note);
        
        const response = await fetch('/teacher/update-attendance', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update UI
            const cell = document.querySelector(`td[data-attendance-id="${attendanceId}"]`);
            if (cell) {
                if (action === 'delete') {
                    cell.classList.remove('present');
                    cell.classList.add('absent');
                    cell.removeAttribute('data-attendance-id');
                    cell.innerHTML = '';
                }
            }
            
            showAlert(data.message, 'success');
        } else {
            showAlert(data.message, 'danger');
        }
    } catch (error) {
        console.error('Error updating attendance record:', error);
        showAlert('Error updating attendance. Please try again.', 'danger');
    } finally {
        hideLoading();
    }
}

// Mark attendance for a student
async function markAttendance(studentId, subjectId, date, note) {
    try {
        showLoading();
        
        const formData = new FormData();
        formData.append('student_id', studentId);
        formData.append('subject_id', subjectId);
        formData.append('date', date);
        formData.append('note', note);
        
        const response = await fetch('/teacher/mark-attendance', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update UI
            const cell = document.querySelector(`td[data-date="${date}"][data-student="${studentId}"][data-subject="${subjectId}"]`);
            if (cell) {
                cell.classList.remove('absent');
                cell.classList.add('present');
                cell.setAttribute('data-attendance-id', data.attendance_id);
                cell.innerHTML = '<i class="fas fa-check"></i>';
            }
            
            showAlert(data.message, 'success');
        } else {
            showAlert(data.message, 'danger');
        }
    } catch (error) {
        console.error('Error marking attendance:', error);
        showAlert('Error marking attendance. Please try again.', 'danger');
    } finally {
        hideLoading();
    }
}

// Send emails to selected students
async function sendEmails() {
    try {
        const selectedStudents = Array.from(document.querySelectorAll('input[name="student_checkbox"]:checked'))
            .map(checkbox => checkbox.value);
            
        if (selectedStudents.length === 0) {
            showAlert('Please select at least one student', 'warning');
            return;
        }
        
        const subject = document.getElementById('emailSubject').value;
        const message = document.getElementById('emailMessage').value;
        
        if (!subject || !message) {
            showAlert('Please provide both subject and message', 'warning');
            return;
        }
        
        showLoading();
        
        const formData = new FormData();
        selectedStudents.forEach(id => formData.append('student_ids', id));
        formData.append('subject', subject);
        formData.append('message', message);
        
        const response = await fetch('/teacher/send-email', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            // Clear form
            document.getElementById('emailSubject').value = '';
            document.getElementById('emailMessage').value = '';
            // Uncheck all checkboxes
            document.querySelectorAll('input[name="student_checkbox"]').forEach(cb => cb.checked = false);
        } else {
            showAlert(data.message, 'danger');
        }
    } catch (error) {
        console.error('Error sending emails:', error);
        showAlert('Error sending emails. Please try again.', 'danger');
    } finally {
        hideLoading();
    }
}

// Helper function to show loading overlay
function showLoading() {
    let overlay = document.querySelector('.loading-overlay');
    
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        overlay.appendChild(spinner);
        
        document.body.appendChild(overlay);
    }
    
    overlay.style.display = 'flex';
}

// Helper function to hide loading overlay
function hideLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// Helper function to show alerts
function showAlert(message, type) {
    const alertContainer = document.getElementById('alert-container');
    if (!alertContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alert);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }
    }, 5000);
}

// Toggle "select all" functionality
function toggleSelectAll() {
    const selectAllCheckbox = document.getElementById('select-all-students');
    if (!selectAllCheckbox) return;
    
    const studentCheckboxes = document.querySelectorAll('input[name="student_checkbox"]');
    
    studentCheckboxes.forEach(checkbox => {
        checkbox.checked = selectAllCheckbox.checked;
    });
}

// Initialize attendance functions when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initAttendanceCalendar();
    
    // Set up select all handler
    const selectAllCheckbox = document.getElementById('select-all-students');
    if (selectAllCheckbox) {
        selectAllCheckbox.addEventListener('change', toggleSelectAll);
    }
    
    // Set up send email handler
    const sendEmailBtn = document.getElementById('sendEmailBtn');
    if (sendEmailBtn) {
        sendEmailBtn.addEventListener('click', sendEmails);
    }
});
