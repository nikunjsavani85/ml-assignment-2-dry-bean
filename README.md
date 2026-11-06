# Machine Learning Assignment 2: Dry Bean Classification

|                    | Link                                                                   |
| ------------------ | ---------------------------------------------------------------------- |
| GitHub repository  | https://github.com/nikunjsavani85/ml-assignment-2-dry-bean             |
| Live Streamlit app | https://ml-assignment-2-dry-bean-gbqbapzsntbesykcs9jgy6.streamlit.app/ |

## a. Problem statement

Dry beans are sorted by variety before sale, and manual sorting is slow and inconsistent. The goal is to predict the variety of a dry bean from 16 shape measurements extracted from its image.

This is a **multiclass classification** problem. Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine) are trained on the same training data, evaluated on the same held-out test set with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and compared. The trained models are saved and served through a Streamlit app, where a user uploads labelled test data, selects a model and sees its metrics, confusion matrix and classification report. The app never retrains the models.

## b. Dataset description

Source: [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), DOI [10.24432/C50S4B](https://doi.org/10.24432/C50S4B), licensed CC BY 4.0. Koklu, M. and Ozkan, I.A. (2020), *Multiclass classification of dry beans using computer vision and machine learning techniques*, Computers and Electronics in Agriculture, 174, 105507.

| Item               | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| Task               | Multiclass classification (7 classes)                            |
| Instances          | 13,611 original; 68 exact duplicates removed; 13,543 used        |
| Input features     | 16 numeric measurements                                          |
| Target column      | `Class`                                                        |
| Classes            | BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA             |
| Missing values     | None                                                             |
| Train / test split | Stratified 80/20, seed 42: 10,834 training rows, 2,709 test rows |

The images of 13,611 grains of seven registered dry bean varieties were taken with a high-resolution camera, segmented, and 16 shape features were computed for each grain (12 dimension features and 4 shape factors).

**Input features (All numeric)**

| Feature                              | Meaning                                                       |
| ------------------------------------ | ------------------------------------------------------------- |
| `Area`                             | Number of pixels inside the bean boundary                     |
| `Perimeter`                        | Length of the bean boundary                                   |
| `MajorAxisLength`                  | Length of the longest line that can be drawn across the bean  |
| `MinorAxisLength`                  | Longest line perpendicular to the major axis                  |
| `AspectRation`                     | Ratio of major to minor axis length (source spelling kept)    |
| `Eccentricity`                     | Eccentricity of the ellipse with the same moments as the bean |
| `ConvexArea`                       | Pixels in the smallest convex polygon containing the bean     |
| `EquivDiameter`                    | Diameter of a circle with the same area as the bean           |
| `Extent`                           | Bean area divided by its bounding-box area                    |
| `Solidity`                         | Bean area divided by its convex area                          |
| `roundness`                        | 4π · Area / Perimeter²                                     |
| `Compactness`                      | EquivDiameter / MajorAxisLength                               |
| `ShapeFactor1` to `ShapeFactor4` | Four further shape ratios derived from the axes and area      |

**Class distribution (after removing duplicates)**

| Class    | Rows  | Share | Rows in test set |
| -------- | ----- | ----- | ---------------- |
| BARBUNYA | 1,322 | 9.8%  | 265              |
| BOMBAY   | 522   | 3.9%  | 104              |
| CALI     | 1,630 | 12.0% | 326              |
| DERMASON | 3,546 | 26.2% | 709              |
| HOROZ    | 1,860 | 13.7% | 372              |
| SEKER    | 2,027 | 15.0% | 406              |
| SIRA     | 2,636 | 19.5% | 527              |

The classes are moderately imbalanced (DERMASON is about 7 times larger than BOMBAY), so macro-averaged Precision, Recall and F1 and MCC are reported alongside Accuracy.

**Preprocessing**

- 68 exact duplicate rows were removed before splitting, preventing identical records from appearing in both training and test data.
- No missing values, so no imputation was needed. All features are numeric, so no encoding was needed.
- Stratified 80/20 train/test split with seed 42 keeps approximately the same class proportions in both parts.
- `StandardScaler` is fitted inside each model pipeline on training data only, which avoids leaking test statistics.

## c. Github Repository Link

https://github.com/nikunjsavani85/ml-assignment-2-dry-bean

Live Streamlit app: https://ml-assignment-2-dry-bean-gbqbapzsntbesykcs9jgy6.streamlit.app/

Repository contents:

```text
ml-assignment-2-dry-bean/
├── streamlit_app.py              Streamlit app (loads saved models, no retraining)
├── classification_models.ipynb   Source code: data preparation, training, evaluation, model export
├── requirements.txt              Pinned dependencies
├── README.md
├── test_data.csv                 Held-out test data (16 features + Class), 2,709 rows
├── data/
│   └── Dry_Bean_Dataset.xlsx     Original UCI workbook
├── model/                        Saved trained models for all five classifiers
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   └── metadata.json             Features, classes, split, versions and model settings
└── results/                      Metrics, cross-validation, classification reports, plots
```

## d. Models used

All five models are scikit-learn pipelines trained on the same split. Logistic Regression, KNN, Naive Bayes (Gaussian) and SVM use `StandardScaler`; the Decision Tree uses raw features.

| Model                  | Settings                                                                                                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression    | C=1, max_iter=2000                                                                                                                                                           |
| Decision Tree          | max_depth=10, min_samples_leaf=3                                                                                                                                             |
| KNN                    | k=7, distance-weighted                                                                                                                                                       |
| Naive Bayes            | GaussianNB, default settings                                                                                                                                                 |
| Support Vector Machine | RBF kernel, C=10, gamma=scale. The saved local model uses probability=True; the lab screenshot includes a calibration import but does not show the fitted SVM configuration. |

### Comparison table (held-out test set, 2,709 rows)

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ---------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression    | 0.9192   | 0.9947 | 0.9307    | 0.9300 | 0.9302 | 0.9023 |
| Decision Tree          | 0.9022   | 0.9747 | 0.9167    | 0.9169 | 0.9166 | 0.8818 |
| KNN                    | 0.9136   | 0.9854 | 0.9268    | 0.9237 | 0.9250 | 0.8956 |
| Naive Bayes            | 0.8970   | 0.9911 | 0.9081    | 0.9082 | 0.9072 | 0.8762 |
| Support Vector Machine | 0.9228   | 0.9911 | 0.9355    | 0.9338 | 0.9346 | 0.9067 |

Precision, Recall and F1 are macro-averaged; AUC is macro one-vs-rest; MCC is the multiclass form. The table above transcribes the four-decimal results visible in the BITS Virtual Lab execution screenshot for student 2026AJ05436. SVM was selected using training-only, three-fold stratified cross-validation before test evaluation.

Run provenance: the local [results/metrics.csv](results/metrics.csv) and saved SVM belong to the earlier run (Accuracy 0.9236, AUC 0.9944, F1 0.9347, MCC 0.9076). The lab run's exported model and full-precision results have not been supplied here. The other four models match the screenshot at four decimal places; the lab SVM and earlier saved SVM must not be treated as the same result.

### Observations

| ML Model Name                     | Observation about model performance                                                                                                                                                                                                                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression               | Accuracy 0.9192 and F1 0.9302 are second to SVM, while AUC 0.9947 is the highest. The training-validation F1 gap is about 0.0020, suggesting limited overfitting in this experiment. It is a competitive linear baseline.                                                                              |
| Decision Tree                     | AUC 0.9747 is the lowest; F1 is 0.9166. Training F1 0.9653 exceeds validation F1 0.9202 by 0.0451, suggesting overfitting despite the depth and leaf-size limits. The saved class report shows SIRA F1 of 0.8330.                                                                                      |
| KNN                               | Accuracy 0.9136, F1 0.9250 and MCC 0.8956 place it in the middle of the comparison. The perfect training F1 reflects zero-distance self-neighbours with distance weighting, not perfect generalisation. Validation F1 is 0.9356.                                                                       |
| Naive Bayes                       | Accuracy 0.8970, F1 0.9072 and MCC 0.8762 are the lowest, despite AUC 0.9911. Related shape measurements may limit the conditional-independence assumption, but these scores do not establish the cause. The saved report shows BARBUNYA recall of 0.7962.                                             |
| Support Vector Machine            | The lab run has the highest Accuracy 0.9228, Precision 0.9355, Recall 0.9338, F1 0.9346 and MCC 0.9067. Its AUC 0.9911 is below Logistic Regression. The lab cross-validation output also selects SVM, with validation F1 0.9431.                                                                      |
| Overall, Winner for your dataset? | **Support Vector Machine**, selected by training-only cross-validation and leading five of the six test metrics in the lab run. Logistic Regression is close behind and has the highest AUC. These results describe one fixed split, not proof that SVM will always outperform the other models. |

## Run locally

Python 3.12.7 and scikit-learn 1.9.0 with the pinned dependencies.

```bash
git clone https://github.com/nikunjsavani85/ml-assignment-2-dry-bean.git
cd ml-assignment-2-dry-bean
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
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
