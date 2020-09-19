# COVIDFact

Analytics of social media discourse related to COVID-19 including trends, topics and trust

Project Organization
------------

    ├── data               <- Data files for use in training
    │
    ├── models             <- Trained and serialized models (pickle files)
    |
    ├── README.md          <- The top-level README for developers using this project
    │
    ├── covid_classify.py  <- Code for training classifier of COVID/non-COVID posts
    │
    ├── covid_predict.py   <- Code for predicting posts as COVID/non-COVID
    │
    ├── extract_phrases.py <- Code for extracting symptom & treatment phrases
    │
    ├── med_non_med.py     <- Code for training classifier of medical/non-medical posts
    |
    ├── reliable_unreliable_model_trainer.py <- Code for training reliable/unreliable model
    │
    ├── requirements.txt   <- For reproducing the analysis environment (run pip install -r requirements.txt)
    │
    └── word2vec_train.py  <- Code for generating COVIDFact embeddings from abstracts

Dependencies
------------
- Part of the code depends on the [Word2vec Twitter Tokens](https://drive.google.com/file/d/1HYCxleAkc1A2Pm-ND_kxU-Jy8Rlrvi46) file (Google Drive link provided due to large size of 4.6GB).
- To install all Python libraries and modules needed, please run `pip install -r requirements.txt`
--------

<p><small>Repository structure partially based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
