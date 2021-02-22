#!/bin/bash

python3 train_default_classifiers.py './data/D1.tsv' 'relevant' 'LR'
python3 train_default_classifiers.py './data/D1.tsv' 'relevant' 'SVM'
python3 train_default_classifiers.py './data/D1.tsv' 'relevant' 'RF'
python3 train_default_classifiers.py './data/D1.tsv' 'relevant' 'NB'
python3 train_default_classifiers.py './data/D2.tsv' 'com_vape' 'LR'
python3 train_default_classifiers.py './data/D2.tsv' 'com_vape' 'SVM'
python3 train_default_classifiers.py './data/D2.tsv' 'com_vape' 'RF'
python3 train_default_classifiers.py './data/D2.tsv' 'com_vape' 'NB'
python3 train_default_classifiers.py './data/D3.tsv' 'pro_vape' 'LR'
python3 train_default_classifiers.py './data/D3.tsv' 'pro_vape' 'SVM'
python3 train_default_classifiers.py './data/D3.tsv' 'pro_vape' 'RF'
python3 train_default_classifiers.py './data/D3.tsv' 'pro_vape' 'NB'
