# Deploying Face Recognition Attendance System on Netlify

This guide will walk you through the steps to deploy this Flask application on Netlify. Since Netlify is primarily designed for static sites, we'll use Netlify Functions to run our Flask application.

## Prerequisites

1. A Netlify account (sign up at [netlify.com](https://www.netlify.com/) if you don't have one)
2. A GitHub account to host your code repository
3. A PostgreSQL database (Supabase or any other provider)

## Step 1: Prepare Your PostgreSQL Database

1. Sign up for a [Supabase](https://supabase.com/) account if you don't have one
2. Create a new project in Supabase
3. Once the project is created, go to Project Settings > Database
4. Copy the "Connection string" with the format: `postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-ID].supabase.co:5432/postgres`
5. Replace `[YOUR-PASSWORD]` with your database password

## Step 2: Push Your Code to GitHub

1. Create a new GitHub repository
2. Initialize Git in your project directory (if not already done):
   ```bash
   git init
   ```
3. Add all files to Git:
   ```bash
   git add .
   ```
4. Commit the changes:
   ```bash
   git commit -m "Initial commit"
   ```
5. Add your GitHub repository as a remote:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   ```
6. Push the code to GitHub:
   ```bash
   git push -u origin main
   ```

## Step 3: Set Up Files for Netlify Deployment

1. Create a `netlify.toml` file in the root directory with the following content:

```toml
[build]
  command = "pip install -r netlify_requirements.txt"
  publish = "static"
  functions = "netlify_functions"

[dev]
  command = "gunicorn --bind 0.0.0.0:5000 main:app"
  port = 5000
  publish = "static"
  
[[redirects]]
  from = "/*"
  to = "/.netlify/functions/app"
  status = 200
```

2. Create a `runtime.txt` file with the Python version:

```
3.11
```

3. Create a directory for Netlify functions:

```bash
mkdir -p netlify_functions
```

4. Create a `netlify_functions/app.js` file to proxy requests to your Flask app:

```javascript
// netlify_functions/app.js
const serverless = require('serverless-http');
const { spawn } = require('child_process');
const express = require('express');
const app = express();

// Start Flask app as a child process
const flaskProcess = spawn('python', ['-m', 'flask', 'run', '--no-debugger']);

flaskProcess.stdout.on('data', (data) => {
  console.log(`Flask stdout: ${data}`);
});

flaskProcess.stderr.on('data', (data) => {
  console.error(`Flask stderr: ${data}`);
});

// Proxy all requests to Flask
app.all('*', async (req, res) => {
  try {
    const response = await fetch(`http://localhost:5000${req.url}`, {
      method: req.method,
      headers: req.headers,
      body: req.method !== 'GET' && req.method !== 'HEAD' ? req.body : undefined,
    });
    
    const body = await response.text();
    res.status(response.status).send(body);
  } catch (error) {
    console.error('Error proxying to Flask:', error);
    res.status(500).send('Internal Server Error');
  }
});

// Export the serverless handler
module.exports.handler = serverless(app);
```

5. Create a `package.json` file for the Node.js dependencies:

```json
{
  "name": "face-recognition-attendance-system",
  "version": "1.0.0",
  "description": "Face Recognition Attendance System",
  "dependencies": {
    "express": "^4.18.2",
    "serverless-http": "^3.2.0"
  }
}
```

## Step 4: Configure Environment Variables on Netlify

1. Go to your Netlify dashboard and create a new site from Git
2. Connect to your GitHub repository
3. In the site settings, go to "Build & deploy" > "Environment variables"
4. Add the following environment variables:
   - `DATABASE_URL`: Your PostgreSQL connection string from Supabase
   - `SESSION_SECRET`: A long, random string for securing session cookies
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_KEY`: Your Supabase anon key

## Step 5: Deploy Your Site

1. In the Netlify dashboard, go to the "Deploys" tab
2. Click on "Trigger deploy" > "Deploy site"
3. Wait for the build and deployment to complete

## Step 6: Set Up Face Recognition Detection

Since Netlify functions have limited runtime, you might need to optimize your face recognition code:

1. Ensure your OpenCV models are included in your repository
2. Add the following environment variable on Netlify:
   - `OPENCV_DISABLE_OPENCL`: Set to `1` to disable OpenCL
   - `FACE_RECOGNITION_MODEL`: Set to `hog` for CPU-based detection

## Step 7: Database Migration

To ensure your database tables are created:

1. Add a `netlify_functions/migrate.js` file:

```javascript
const { exec } = require('child_process');

exports.handler = async function(event, context) {
  return new Promise((resolve, reject) => {
    exec('python -c "from app import db; db.create_all()"', (error, stdout, stderr) => {
      if (error) {
        console.error(`Error: ${error}`);
        return reject({ statusCode: 500, body: JSON.stringify({ error: error.message }) });
      }
      console.log(`stdout: ${stdout}`);
      console.error(`stderr: ${stderr}`);
      resolve({ statusCode: 200, body: JSON.stringify({ message: 'Migration completed successfully' }) });
    });
  });
};
```

2. Call this function after deployment by visiting: `https://your-netlify-site.netlify.app/.netlify/functions/migrate`

## Troubleshooting

### Database Connection Issues

If you're having trouble connecting to the database:

1. Double-check your DATABASE_URL format
2. Ensure you've allowed all IP addresses in Supabase network settings
3. Check that you've replaced the placeholder password in the connection string

### Face Recognition Not Working

If face recognition isn't working:

1. Check if the haarcascades directory is included in your deployment
2. Consider using a simpler face detection method
3. Ensure all dependencies are correctly installed

### Deployment Timeouts

If your builds are timing out:

1. Optimize your requirements by removing unnecessary packages
2. Split heavyweight operations into separate functions
3. Consider moving face recognition processing to a separate service

## Important Notes

- Netlify Functions have a 10-second execution limit, which might be challenging for face recognition tasks
- Consider using Netlify's paid plans for increased function timeout limits
- For a production environment with heavy usage, you might want to explore other hosting options like Heroku, AWS, or GCP

## Need Help?

If you encounter any issues with your deployment, feel free to:

1. Check the Netlify documentation: [https://docs.netlify.com/](https://docs.netlify.com/)
2. Visit the Netlify Community forums: [https://community.netlify.com/](https://community.netlify.com/)
3. Open an issue in your GitHub repository