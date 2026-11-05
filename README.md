# Machine Learning Assignment 2: Dry Bean Classification

## a. Problem statement

Classify dry beans into one of seven varieties using 16 shape measurements extracted from images. Five classification models are trained and compared on the same dataset and train/test split, and the trained models are served through a Streamlit app that evaluates them on an uploaded, labelled test CSV without retraining.

## b. Dataset description

Source: [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), DOI [10.24432/C50S4B](https://doi.org/10.24432/C50S4B), licensed CC BY 4.0. Koklu, M. and Ozkan, I.A. (2020), *Multiclass classification of dry beans using computer vision and machine learning techniques*, Computers and Electronics in Agriculture, 174, 105507.

| Item | Description |
| --- | --- |
| Task | Multiclass classification (7 classes) |
| Instances | 13,611 original; 68 exact duplicates removed; 13,543 used |
| Input features | 16 numeric measurements |
| Target column | `Class` |
| Classes | BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| Missing values | None |
| Train / test split | Stratified 80/20, seed 42: 10,834 training rows, 2,709 test rows |

Features: `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4` (source spelling kept, including `AspectRation`).

Duplicates were removed before splitting so identical rows cannot appear on both sides. Scaling is fitted inside each model pipeline on training data only.

## c. Github Repository Link

https://github.com/nikunjsavani85/ml-assignment-2-dry-bean

Live Streamlit app: *to be added after deployment*

## d. Models used

All five models are scikit-learn pipelines trained on the same split. Logistic Regression, KNN, Naive Bayes (Gaussian) and SVM use `StandardScaler`; the Decision Tree uses raw features.

| Model | Settings |
| --- | --- |
| Logistic Regression | C=1, max_iter=2000 |
| Decision Tree | max_depth=10, min_samples_leaf=3 |
| KNN | k=7, distance-weighted |
| Naive Bayes | GaussianNB, default settings |
| Support Vector Machine | RBF kernel, C=10, gamma=scale, probability estimates enabled |

### Comparison table (held-out test set, 2,709 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9192 | 0.9947 | 0.9307 | 0.9300 | 0.9302 | 0.9023 |
| Decision Tree | 0.9022 | 0.9747 | 0.9167 | 0.9169 | 0.9166 | 0.8818 |
| KNN | 0.9136 | 0.9854 | 0.9268 | 0.9237 | 0.9250 | 0.8956 |
| Naive Bayes | 0.8970 | 0.9911 | 0.9081 | 0.9082 | 0.9072 | 0.8762 |
| Support Vector Machine | 0.9236 | 0.9944 | 0.9356 | 0.9340 | 0.9347 | 0.9076 |

Precision, Recall and F1 are macro-averaged; AUC is macro one-vs-rest; MCC is the multiclass form. Full-precision values are in [results/metrics.csv](results/metrics.csv).

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Second-best F1 (0.9302) and the highest AUC (0.9947). The small gap between training and validation F1 (0.0020) shows it generalises well; a strong linear baseline. |
| Decision Tree | Lowest AUC (0.9747) and F1 of 0.9166. Its training-validation F1 gap (0.0451) indicates some overfitting despite the depth and leaf-size limits. |
| KNN | F1 of 0.9250. Benefits from feature scaling since it is distance-based; its perfect training score comes from each point being its own neighbour, not perfect generalisation. |
| Naive Bayes | Lowest accuracy (0.8970) and F1 (0.9072), yet a high AUC (0.9911). The independence assumption is restrictive for strongly correlated shape features such as Area, Perimeter and ConvexArea. |
| Support Vector Machine | Best accuracy (0.9236), F1 (0.9347) and MCC (0.9076). It also had the best 3-fold cross-validation F1 on training data (0.9431). |
| Overall, Winner for your dataset? | **Support Vector Machine.** It was chosen by cross-validation on training data before looking at the test set, and it also led on test accuracy, F1 and MCC. Logistic Regression is a close second with the best AUC. |

## Repository structure

```text
ml-assignment-2-dry-bean/
├── app.py                        Streamlit app (loads saved models, no retraining)
├── classification_models.ipynb   Data preparation, training, evaluation, model export
├── requirements.txt
├── README.md
├── test_data.csv                 Held-out test set (16 features + Class)
├── data/
│   └── Dry_Bean_Dataset.xlsx     Original UCI workbook
├── model/                        Five trained pipelines + metadata.json
└── results/                      Metrics, classification reports, plots
```

## Run locally

Python 3.11 is recommended (the models were trained with Python 3.11.9 and scikit-learn 1.9.0).

```bash
git clone https://github.com/nikunjsavani85/ml-assignment-2-dry-bean.git
cd ml-assignment-2-dry-bean
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, upload `test_data.csv` from the sidebar, and choose a model.

To retrain, open `classification_models.ipynb` and run all cells. This regenerates `model/`, `test_data.csv` and `results/`.

## Streamlit app

- CSV upload for labelled test data, with a downloadable sample `test_data.csv`.
- Model selection dropdown for the five classifiers.
- Six metric cards: Accuracy, AUC, Precision, Recall, F1, MCC.
- Confusion matrix and classification report for the selected model.
- Comparison table of all five models on the uploaded file.
- Upload validation: missing or unexpected columns, non-numeric values, empty cells and unknown class labels are reported clearly. AUC is shown as n/a if the file does not contain all seven classes.
