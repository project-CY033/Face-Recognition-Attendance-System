# Render Deployment Guide for Face Recognition Attendance System

This guide provides the steps to deploy this Flask application to Render.com.

## Prerequisites

- A Render.com account
- Your code pushed to a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

### 1. Create the Deployment Files

Make sure these files exist in your project root:

#### build.sh
This script will run during the build phase to set up the environment:

```bash
#!/usr/bin/env bash
# Exit on error
set -o errexit

# Create required directories
mkdir -p uploads
mkdir -p instance

# Create OpenCV haarcascades directory and download required files
mkdir -p haarcascades
cd haarcascades
wget -q https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
wget -q https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
cd ..

# Make build script executable
chmod +x build.sh

echo "Build completed successfully"
```

Don't forget to make the script executable:
```bash
chmod +x build.sh
```

#### render.yaml (Optional)
This file provides Render with configuration details:

```yaml
services:
  # Web service
  - type: web
    name: face-recognition-attendance
    env: python
    buildCommand: ./build.sh
    startCommand: gunicorn --bind 0.0.0.0:$PORT --reuse-port main:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.6
      - key: PORT
        value: 10000
```

### 2. Connect Render to Your Repository

1. Log in to your Render account
2. Click on "New +" and select "Web Service"
3. Connect your Git repository
4. Configure the following settings:
   - **Name**: Choose a name for your application
   - **Environment**: Python
   - **Region**: Choose the closest region to your users
   - **Branch**: main (or your default branch)
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT main:app`

### 3. Set Up a Database (Two Options)

#### Option 1: Use Render's PostgreSQL Database (Recommended)
1. In your Render dashboard, click on "New +" and select "PostgreSQL"
2. Configure your database settings
3. Once created, Render will provide you with database credentials
4. Use these credentials in your environment variables

#### Option 2: Use Supabase (Optional)
1. Create a Supabase project if you haven't already
2. Get your connection string from Supabase dashboard > Settings > Database > Connection string
3. Use this as your DATABASE_URL in environment variables

### 4. Configure Environment Variables

Add the following environment variables in the Render dashboard:

**Required Variables:**
- `PORT`: 10000 (Render will automatically map this to the correct port)
- `SESSION_SECRET`: A secure random string for your application sessions

**Database Variables (Option 1 - Render PostgreSQL):**
- `DATABASE_URL`: This will be automatically set if you use Render's PostgreSQL
OR
- `PGUSER`, `PGPASSWORD`, `PGHOST`, `PGPORT`, `PGDATABASE`: These will be provided by Render

**Supabase Variables (Option 2 - Optional):**
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key

### 5. Deploy

Click "Create Web Service" and Render will begin deploying your application.

## Database Handling in the Application

The application is designed to be flexible with database connections:

1. It first tries to use the `DATABASE_URL` environment variable
2. If that's not available, it looks for individual PostgreSQL variables (`PGUSER`, etc.)
3. It also checks for Supabase credentials if direct PostgreSQL isn't available
4. As a last resort, it falls back to SQLite for local development

This approach ensures your application can run in various environments with minimal configuration changes.

## Troubleshooting Deployment Issues

### Database Connection Issues

If you see database connection errors:

1. Check that your database credentials are correctly set in environment variables
2. For Render PostgreSQL, ensure you've created the database service and linked it
3. For Supabase, verify that external connections are allowed and credentials are correct
4. Try using the individual PostgreSQL variables instead of DATABASE_URL

### Missing haarcascades Directory

If face recognition fails with "Could not find haarcascades directory":

- Ensure the build.sh script correctly downloads the cascade files
- Check that the application can access the haarcascades directory

### Port Binding Issues

If Render shows "No open ports detected":

- Make sure you're binding to the PORT environment variable provided by Render
- The correct format is: `gunicorn --bind 0.0.0.0:$PORT main:app`