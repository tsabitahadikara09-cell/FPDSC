# DermaCNN — Streamlit Cloud Demo

Mini project demo for ISIC-style skin lesion image processing (MEL, NV, BKL).\n

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset layout

Images are read automatically from `dataset/images/<CLASS>/`.

- `dataset/images/MEL/`
- `dataset/images/NV/`
- `dataset/images/BKL/`

No Windows paths or local virtual environment are required by Streamlit Cloud.

## Deploy

Push the project to GitHub, then create a Streamlit Community Cloud app with `app.py` as the entry point.

## Important limitation

The included dataset has only one example image per class. Augmented training is only a software demonstration. The resulting metrics are not a valid estimate of generalization, diagnostic accuracy, or clinical performance. Replace the sample images with a properly labeled and separated dataset before conducting research or reporting results.
