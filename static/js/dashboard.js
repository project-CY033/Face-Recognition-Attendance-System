/**
 * Dashboard functionality
 */

// Initialize dashboard components
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setupFilterListeners();
    setupNotificationHandlers();
});

// Initialize charts for dashboard
function initCharts() {
    // Attendance chart (if present)
    const attendanceChartElem = document.getElementById('attendanceChart');
    if (attendanceChartElem) {
        const ctx = attendanceChartElem.getContext('2d');
        
        // Get chart data from the data attributes
        const labels = JSON.parse(attendanceChartElem.getAttribute('data-labels') || '[]');
        const values = JSON.parse(attendanceChartElem.getAttribute('data-values') || '[]');
        
        // Create chart using Chart.js
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Attendance %',
                    data: values,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y + '%';
                            }
                        }
                    }
                }
            }
        });
    }
}

// Set up filter listeners for tables
function setupFilterListeners() {
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('keyup', filterTable);
    }
    
    const filterSelects = document.querySelectorAll('select[data-filter-target]');
    filterSelects.forEach(select => {
        select.addEventListener('change', applyFilters);
    });
}

// Filter table based on search input
function filterTable() {
    const searchInput = document.getElementById('search-input');
    const filterText = searchInput.value.toLowerCase();
    const tableBody = document.querySelector(searchInput.getAttribute('data-table-target'));
    
    if (!tableBody) return;
    
    const rows = tableBody.getElementsByTagName('tr');
    
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        const cells = row.getElementsByTagName('td');
        let shouldShow = false;
        
        for (let j = 0; j < cells.length; j++) {
            const cellText = cells[j].textContent || cells[j].innerText;
            
            if (cellText.toLowerCase().indexOf(filterText) > -1) {
                shouldShow = true;
                break;
            }
        }
        
        row.style.display = shouldShow ? '' : 'none';
    }
}

// Apply filters to a table
function applyFilters() {
    const filterSelects = document.querySelectorAll('select[data-filter-target]');
    const filters = {};
    
    // Collect filter values
    filterSelects.forEach(select => {
        const filterName = select.getAttribute('data-filter-name');
        const filterValue = select.value;
        
        if (filterValue !== 'all') {
            filters[filterName] = filterValue;
        }
    });
    
    // Apply filters to table rows
    const tableBody = document.querySelector(filterSelects[0].getAttribute('data-filter-target'));
    if (!tableBody) return;
    
    const rows = tableBody.getElementsByTagName('tr');
    
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        let shouldShow = true;
        
        // Check each filter
        for (const [filterName, filterValue] of Object.entries(filters)) {
            const cellValue = row.getAttribute(`data-${filterName}`);
            
            if (cellValue !== filterValue) {
                shouldShow = false;
                break;
            }
        }
        
        row.style.display = shouldShow ? '' : 'none';
    }
}

// Setup notification handlers
function setupNotificationHandlers() {
    const notificationItems = document.querySelectorAll('.notification-item');
    
    notificationItems.forEach(item => {
        item.addEventListener('click', () => {
            const notificationId = item.getAttribute('data-notification-id');
            if (notificationId) {
                markNotificationRead(notificationId);
                item.classList.remove('unread');
            }
        });
    });
}

// Mark notification as read
async function markNotificationRead(notificationId) {
    try {
        const response = await fetch(`/mark-notification-read/${notificationId}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            console.error('Failed to mark notification as read');
        }
    } catch (error) {
        console.error('Error marking notification as read:', error);
    }
}
