# GitHub Setup Instructions

## Step 1: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Repository name: `TheBiasLab-Final-Round` (or your preferred name)
5. Description: `Advanced Media Bias Detection Engine - The Bias Lab Final Round`
6. Make it **Public** (so reviewers can access it)
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add the remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/TheBiasLab-Final-Round.git

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify Upload

1. Refresh your GitHub repository page
2. You should see all your files uploaded
3. **Important**: Verify that `config.env` is NOT visible (it should be hidden by .gitignore)
4. Check that the README.md displays correctly

## Example Commands (replace YOUR_USERNAME)

```bash
# If your GitHub username is "johnsmith":
git remote add origin https://github.com/johnsmith/TheBiasLab-Final-Round.git
git branch -M main
git push -u origin main
```

## Security Check

After pushing, verify these files are NOT visible on GitHub:
- ❌ `config.env` (contains API keys)
- ❌ `venv/` (virtual environment)
- ❌ `__pycache__/` (Python cache files)

These files should be visible:
- ✅ `README.md`
- ✅ `app/` folder with all Python files
- ✅ `requirements.txt`
- ✅ `Dockerfile`
- ✅ `main.py`

## Troubleshooting

If you get permission errors:
1. Make sure you're logged into GitHub
2. Consider using SSH instead of HTTPS
3. Or use GitHub CLI: `gh repo create TheBiasLab-Final-Round --public --push`

## Final Step

Once pushed successfully:
1. Copy the GitHub repository URL
2. Share it with The Bias Lab reviewers
3. The live demo will be at: `https://your-deployment-url.com`
