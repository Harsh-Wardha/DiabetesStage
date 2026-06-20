# Vercel deployment

1. Prepare the real three-class dataset locally.
2. Run `python train_models.py`; never train on Vercel.
3. Run `python scripts/export_report_assets.py`, `python scripts/verify_artifacts.py`, `pytest`, and `python scripts/check_deployment_size.py`.
4. Commit all three `models/*.joblib`, `models/model_registry.json`, and `artifacts/*.json` files.
5. Confirm raw/processed datasets, the root CSV, tests, and reports are excluded by `.vercelignore`.
6. Push to GitHub and import the repository into Vercel.
7. Let current zero-configuration Flask detection import `app.py`; no legacy `builds`, `routes`, or `@vercel/python` configuration is used.
8. Set `SECRET_KEY` to a long random value and `FLASK_ENV=production`.
9. Deploy; open `/health`, then test `/`, `/predict`, `/dashboard`, `/insights`, and `/methodology`.

CLI deployment:

```text
npm install -g vercel
vercel login
vercel
vercel --prod
```

Production reads committed pretrained artifacts. It does not train, fetch data, store submissions, write runtime files, or call an external prediction API.
