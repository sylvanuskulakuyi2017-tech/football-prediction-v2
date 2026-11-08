# Deploy on Railway

After pushing this release to GitHub and passing its checks:

1. In Railway, create a project from the GitHub repository football-prediction-v2.
2. Use this repository's root as the source directory. Railway detects the root
   Dockerfile. Leave build/start command overrides empty so its CMD is used.
3. Under the service deployment settings, set the healthcheck path to
   /_stcore/health and allow a 300-second healthcheck timeout.
4. Deploy, then use the service networking settings to generate a public domain.
5. Open the domain and check Match scenario, Historical replay and Model performance.
   Compare the displayed model/data cutoffs and benchmark with the local app.

The startup script verifies dependency versions, asset hashes and one saved
forecast before launching Streamlit. It listens on 0.0.0.0 and Railway's PORT
variable, with a local fallback of 8080. No database, volume or API key is required
for this snapshot-based version. The app exports predictions through downloads.

The files use the Dockerfile deployment route. Railway-specific healthcheck
settings are entered in the dashboard; this release does not depend on the legacy
railway.json/railway.toml configuration route. Check the platform's current usage
and billing settings in your own account before choosing deployment resources.

## Optional local Docker check

If Docker is available, from the repository root:

```bash
docker build -t football-prediction-v2 .
docker run --rm -p 8501:8501 -e PORT=8501 football-prediction-v2
```

Open http://localhost:8501. A successful local Streamlit check alone does not
establish that a Docker build or Railway deployment has succeeded. The exporter
does not start Docker or create cloud resources. GitHub Actions runs the build
after you push the repository.

## Troubleshooting

- A dependency error: restore requirements.lock.txt; do not upgrade scikit-learn
  independently of the saved model.
- An asset/hash error: restore the matching committed files. Do not bypass the
  integrity check by editing a recorded hash.
- A failed healthcheck: inspect startup logs and verify the configured healthcheck
  path and PORT. Streamlit's health endpoint checks the server; manually exercising
  the app remains part of deployment verification.

References:
- https://docs.railway.com/builds/dockerfiles
- https://docs.railway.com/deployments/healthchecks
- https://docs.railway.com/networking/public-networking
- https://docs.streamlit.io/deploy/tutorials/docker
