# EduPredict AI — Student Performance Predictor

A polished end-to-end ANN web application. It predicts a student's final exam score from six inputs.

## Run locally

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python training/train.py
python -m uvicorn backend.app:app --reload
```

Open: http://127.0.0.1:8000

The frontend and API are served by the same FastAPI application.

## API

- `GET /health`
- `POST /predict`
- `GET /docs` for API documentation

## Note

The included CSV is a small educational dataset. For a public portfolio deployment, replace it with a larger, real-world dataset and retrain/evaluate the model.
