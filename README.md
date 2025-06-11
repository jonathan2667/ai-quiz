# AI Learning Platform - Deployment Guide

This AI learning platform was originally created for personal use. If you need help to host this, you can contact me via LinkedIn anytime to set up this nice ranking website for AI.

If you want to host your own version, follow these simple deployment steps using Netlify (free) and Firebase database.

## 🚀 Deployment Steps

### Step 1: Deploy on Netlify

1. **Create a Netlify Account**
   - Go to [netlify.com](https://netlify.com) and sign up for free
   - Connect your GitHub account

2. **Deploy the Project**
   - Fork or clone this repository to your GitHub account
   - In Netlify dashboard, click "New site from Git"
   - Choose GitHub and select your forked repository
   - Set build settings:
     - Build command: (leave empty)
     - Publish directory: (leave empty or set to root `/`)
   - Click "Deploy site"

3. **Your Site is Live!**
   - Netlify will provide you with a free URL (e.g., `https://your-site-name.netlify.app`)
   - The site will auto-deploy whenever you push changes to your GitHub repository

### Step 2: Add Firebase API Keys

1. **Create Firebase Project**
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Click "Create a project"
   - Follow the setup wizard
   - Enable Firestore Database (start in test mode)

2. **Get Your Firebase Configuration**
   - In Firebase Console, go to Project Settings (gear icon)
   - Scroll down to "Your apps" section
   - Click "Add app" → Web app (</>) 
   - Register your app and copy the configuration object

3. **Add Firebase Config to Your Project**
   - **IMPORTANT**: Replace the placeholder API keys in `user-counter.js` file (lines 6-15) with your real Firebase configuration
   - In the file `user-counter.js`, find the `fallbackConfig` object and replace all placeholder values:
   ```javascript
   // In user-counter.js - Replace these placeholder values:
   const fallbackConfig = {
     apiKey: "YOUR_API_KEY_HERE", // ← Replace with your real API key
     authDomain: "your-project.firebaseapp.com", // ← Replace with your domain
     databaseURL: "https://your-project-default-rtdb.region.firebasedatabase.app", // ← Replace with your database URL
     projectId: "your-project-id", // ← Replace with your project ID
     storageBucket: "your-project.firebasestorage.app", // ← Replace with your storage bucket
     messagingSenderId: "123456789", // ← Replace with your sender ID
     appId: "1:123456789:web:abcdefghijklmnop", // ← Replace with your app ID
     measurementId: "G-XXXXXXXXXX" // ← Replace with your measurement ID
   };
   ```

4. **Important Security Notes**
   - **Before pushing to GitHub**: Remove any fallback/demo API keys from the code
   - Set Firebase config to default/empty values in public repository
   - Add `firebase-config.js` to your `.gitignore` file
   - Use Netlify environment variables for production deployment

5. **Push to Git**
   ```bash
   git add .
   git commit -m "Add Firebase configuration"
   git push origin main
   ```

## ⚠️ Security Reminder

- Never commit real API keys to public repositories
- Use environment variables for sensitive configuration
- Set up proper Firebase security rules for production use

## 🎯 That's It!

Your AI learning platform is now live and ready to use with your own Firebase database for session tracking and user data storage.
