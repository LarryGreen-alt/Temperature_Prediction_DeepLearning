# Live forecast demo

FastAPI backend + a single vanilla-JS page. Pick a city and a model, hit
Predict, and see a 24-hour forecast built from live Open-Meteo data.

## Run it

From the repo root, with the project's venv active:

```
pip install -r demo/requirements.txt
uvicorn demo.main:app --reload
```

Then open http://localhost:8000

Model loading happens once at startup and takes a few seconds; the page
will work as soon as the terminal prints "Both models loaded."
