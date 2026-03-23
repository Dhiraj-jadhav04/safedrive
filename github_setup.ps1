# SafeDrive GitHub Upload Script
# This script will initialize a Git repository and help you push it to your account.

$repo_name = "safedrive"
$user_name = "Dhiraj-jadhav04"
$remote_url = "https://github.com/$user_name/$repo_name.git"

Write-Host "🚀 Preparing to upload SafeDrive to GitHub..." -ForegroundColor Cyan

# 1. Check for Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Git is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/downloads" -ForegroundColor Yellow
    exit
}

# 2. Local Initialization
Write-Host "📦 Initializing local repository..."
git init
git add .
git commit -m "🚀 Initial Commit: SafeDrive AI Fatigue Monitor - Ready for Deployment"

# 3. Remote Setup
Write-Host "🔗 Adding remote: $remote_url"
git remote add origin $remote_url
git branch -M main

# 4. Final Instructions
Write-Host "--------------------------------------------------------" -ForegroundColor Green
Write-Host "✅ Local repository is ready!" -ForegroundColor Green
Write-Host "--------------------------------------------------------"
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com/new"
Write-Host "2. Create a PUBLIC repository named: $repo_name"
Write-Host "3. DO NOT initialize it with README or gitignore (I have already created them)."
Write-Host "4. Once created, run the following command in this terminal:" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan
Write-Host "--------------------------------------------------------"
