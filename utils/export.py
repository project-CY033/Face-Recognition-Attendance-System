import io
import csv
from flask import send_file
import xlsxwriter
from fpdf import FPDF

def generate_excel(data, filename):
    """
    Generate Excel file from data
    
    Args:
        data (list): List of rows (each row is a list of values)
        filename (str): Base filename without extension
    
    Returns:
        Response: Flask response with Excel file
    """
    # Create an in-memory output file
    output = io.BytesIO()
    
    # Create Excel workbook and worksheet
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    
    # Add header formatting
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D3D3D3',
        'border': 1
    })
    
    # Add cell formatting
    cell_format = workbook.add_format({
        'border': 1
    })
    
    # Write data to worksheet
    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            if row_idx == 0:  # Header row
                worksheet.write(row_idx, col_idx, value, header_format)
            else:
                worksheet.write(row_idx, col_idx, value, cell_format)
    
    # Set column widths
    worksheet.set_column(0, 0, 5)  # S.No
    worksheet.set_column(1, 1, 25)  # Name
    worksheet.set_column(2, 2, 15)  # College ID
    worksheet.set_column(3, 3, 15)  # Phone
    worksheet.set_column(4, 4, 25)  # Email
    worksheet.set_column(5, len(data[0]), 12)  # Dates
    
    # Close the workbook
    workbook.close()
    
    # Seek to start of file
    output.seek(0)
    
    # Return the file
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f"{filename}.xlsx"
    )

def generate_pdf(data, filename, subject_name):
    """
    Generate PDF file from data
    
    Args:
        data (list): List of rows (each row is a list of values)
        filename (str): Base filename without extension
        subject_name (str): Subject name for the header
    
    Returns:
        Response: Flask response with PDF file
    """
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Attendance Report - {subject_name}", 0, 1, 'C')
    pdf.ln(10)
    
    # Set font back to normal
    pdf.set_font("Arial", size=10)
    
    # Calculate column widths
    page_width = pdf.w - 2 * pdf.l_margin
    col_count = len(data[0])
    
    # Adjust col_widths based on content type
    col_widths = [15]  # S.No (small)
    col_widths.append(40)  # Name (larger)
    col_widths.append(30)  # College ID (medium)
    col_widths.append(30)  # Phone (medium)
    col_widths.append(40)  # Email (larger)
    
    # Calculate remaining width for date columns
    remaining_width = page_width - sum(col_widths)
    date_col_width = remaining_width / (col_count - 5) if col_count > 5 else 0
    
    for _ in range(col_count - 5):
        col_widths.append(date_col_width)
    
    # Add header row
    pdf.set_fill_color(200, 200, 200)  # Light gray background
    for i, col_name in enumerate(data[0]):
        pdf.cell(col_widths[i], 10, str(col_name), 1, 0, 'C', True)
    pdf.ln()
    
    # Add data rows
    for row in data[1:]:
        # Check if we need to add a new page
        if pdf.get_y() + 10 > pdf.page_break_trigger:
            pdf.add_page()
            # Repeat header
            pdf.set_font("Arial", 'B', 10)
            for i, col_name in enumerate(data[0]):
                pdf.cell(col_widths[i], 10, str(col_name), 1, 0, 'C', True)
            pdf.ln()
            pdf.set_font("Arial", size=10)
        
        # Add row data
        for i, cell_value in enumerate(row):
            pdf.cell(col_widths[i], 10, str(cell_value), 1, 0, 'L')
        pdf.ln()
    
    # Generate in-memory file
    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output())
    pdf_output.seek(0)
    
    # Return the file
    return send_file(
        pdf_output,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"{filename}.pdf"
    )
