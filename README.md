# Movie Recommender Project

A simple movie recommendation system built with Python and Streamlit.

This project uses the TMDB 5000 datasets to generate recommendations with two methods:
- Content-based similarity
- Metadata-based collaborative similarity (no user ratings file is used)

## Features

- Train recommendation artifacts from CSV files
- Choose a movie from a dropdown
- Switch between recommendation methods
- Show top 5 recommended movies
- Display basic movie details (genres and overview)

## Project Structure

- `train_model.py` builds and saves model artifacts
- `app.py` runs the Streamlit web app
- `tmdb_5000_movies.csv` movie metadata dataset
- `tmdb_5000_credits.csv` credits dataset
- `movie_list.pkl` processed movie list (generated)
- `similarity.pkl` similarity matrices (generated)
- `requirements.txt` Python dependencies

## Requirements

- Python 3.10+ (project currently appears to be running in a Python virtual environment)
- pip

## Setup

### 1. Clone or open the project folder

Open this folder in VS Code:

`Movie Recommender Project`

### 2. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv movie_env
.\movie_env\Scripts\Activate.ps1
```

If you already have the environment, just activate it:

```powershell
.\movie_env\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Train the Recommendation Artifacts

Run training to regenerate `movie_list.pkl` and `similarity.pkl`:

```powershell
python train_model.py
```

Expected output includes something like:
- `Saved: movie_list.pkl, similarity.pkl`
- `Movies processed: <number>`

## Run the Streamlit App

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How to Use

1. Select a movie title from the dropdown.
2. Choose recommendation method:
   - Content-Based
   - Collaborative
3. Click **Recommend**.
4. View 5 recommended movies with overview and genres.

## Notes

- If `movie_list.pkl` or `similarity.pkl` is missing, run `python train_model.py` first.
- The collaborative option here is based on metadata co-occurrence (genres/cast/crew), not user ratings.
- The app currently uses local CSV and PKL files from the project root.

## Troubleshooting

- `ModuleNotFoundError`: install packages again with `pip install -r requirements.txt`.
- NLTK-related issues: ensure `nltk` is installed from `requirements.txt`.
- PowerShell script execution blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

## License

MIT License
