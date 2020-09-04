# COVIDFact

Analytics of social media discourse related to COVID-19 including trends, topics and trust

Project Organization
------------

    ├── data               <- Data files for use in training
    │
    ├── models             <- Trained and serialized models (pickle files)
    
    ├── README.md          <- The top-level README for developers using this project
    │
    ├── covid_classify.py  <- Code for training classifier of COVID/non-COVID posts
    │
    ├── covid_predict.py   <- Code for predicting posts as COVID/non-COVID
    │
    ├── extract_phrases.py <- Code for extracting symptom & treatment phrases
    │
    ├── med_non_med.py     <- Code for training classifier of medical/non-medical posts
    │
    ├── requirements.txt   <- For reproducing the analysis environment
    │
    ├── setup.py           <- Makes project pip installable
    │
    └── word2vec_train.py  <- Code for generating COVIDFact embeddings from abstracts
    
--------

<p><small>Project partially based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>