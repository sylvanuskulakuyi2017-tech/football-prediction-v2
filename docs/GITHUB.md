# Publish the release on GitHub

Use the exported release folder as the repository root. The larger working project
and any parent .git folder are not part of this export. Review README.md, LICENSE
and DATA_NOTICE.md. The included MIT licence names Sylvanus Kulakunyi as author.

Create an empty GitHub repository named football-prediction-v2. Do not add another
README, licence or .gitignore when creating it. Then, inside this release folder:

```bash
git init -b main
git add .
git status --short
git commit -m "Add International Match Lab with frozen benchmark"
```

Copy the exact repository URL from GitHub. Add it with git remote add origin,
then run git push -u origin main. GitHub handles authentication through your own
Git credential manager or GitHub CLI; no credentials belong in project files.

The GitHub Actions workflow installs the lock, checks the frozen assets, reconstructs
one saved forecast, loads all three app views, and builds the Docker image.
Passing CI checks functionality and buildability; it does not establish forecast
quality beyond the saved evaluation.

GitHub blocks individual files above 100 MiB. The exporter checks this limit and
prints the largest file. The supplied .gitattributes preserves hashed CSV/JSON bytes.
Do not format or edit frozen report/data files without deliberately creating a new
version with a matching model and manifest.

Reference: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
