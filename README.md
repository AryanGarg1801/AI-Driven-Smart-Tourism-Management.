# AI-Driven Smart Tourism Management System

An intelligent tourism management project developed using Machine Learning to predict tourist expenditure, identify tourist segments, and generate personalized travel recommendations across India.

## Project Features

1. **Tourist Expenditure Prediction**
   - CatBoost regression model.
   - Expenditure prediction using household, trip and destination characteristics.
   - Survey-weighted model evaluation.

2. **Tourist Segmentation**
   - MiniBatch K-Means clustering.
   - Behavioural segmentation based on tourist and trip characteristics.
   - Cluster profiles and visualizations.

3. **Personalized Travel Recommendations**
   - Historical travel-pattern analysis.
   - Similarity-based recommendations.
   - Curated Indian tourist attraction catalogue.

4. **Interactive Web Application**
   - Streamlit interface.
   - Expenditure prediction.
   - Tourist segmentation.
   - Destination discovery and recommendations.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- CatBoost
- TensorFlow / Keras
- Streamlit
- Plotly
- Matplotlib and Seaborn

## Project Structure

```text
Smart_Tourism_Management/
|
|-- Dataset/
|   |-- india_tourist_destinations.csv
|
|-- src/
|   |-- prepare_dataset.py
|   |-- prepare_for_eda.py
|   |-- decode_categories.py
|   |-- validate_dataset.py
|   |-- prepare_model_data.py
|   |-- train_models.py
|   |-- train_models_advanced.py
|   |-- train_catboost.py
|   |-- train_ann1.py
|   |-- train_ann_improved.py
|   |-- finalize_segmentation.py
|   |-- recommendation_engine.py
|   |-- final_model_evaluation.py
|
|-- outputs/
|   |-- eda/
|   |-- models/
|   |-- final_evaluation/
|   |-- segmentation/
|   |-- recommendations/
|
|-- app.py
|-- requirement.txt
|-- README.md
```

## Dataset Information

The expenditure prediction and historical tourism analysis use the Ministry of Statistics and Programme Implementation (MoSPI), Government of India, Domestic Tourism Expenditure Survey (DTES), 80th Round.

The project dataset was prepared using overnight trip information and corresponding expenditure information.

The public repository does not redistribute the restricted raw survey microdata or record-level processed datasets.

Official dataset catalogue:
https://microdata.gov.in/nada/index.php/catalog/307

Users must obtain authorized access to the survey data and comply with its applicable access and redistribution conditions.

The repository includes a curated Indian tourist attraction catalogue.

## Model Evaluation

The CatBoost expenditure model achieved the following test results:

| Metric | Result |
|---|---:|
| R² | 0.3394 |
| Weighted R² | 0.3563 |
| MAE | ₹11,442.68 |
| RMSE | ₹34,846.72 |
| Weighted MAE | ₹9,770.17 |
| Weighted RMSE | ₹30,039.57 |

The advanced Random Forest model achieved an R² of 0.3097 and RMSE of ₹35,620.17 on its test evaluation.

The models demonstrate moderate predictive performance. Expenditure prediction remains challenging because of substantial variation in tourist spending.

The selected behavioural segmentation experiment produced three clusters, with a silhouette score of 0.1112. The relatively low score indicates overlapping tourist segments, so clusters should be interpreted as broad behavioural groups rather than sharply separated categories.

## Installation

Create and activate a virtual environment:

```bash
py -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
py -m pip install -r requirement.txt
```

## Running the Application

```bash
py -m streamlit run app.py
```

**Data requirement:** The application needs the derived DTES dataset files locally. After obtaining authorized survey access, place the required source CSV files in `Dataset/DTES25_26_CSV/` and run the relevant preparation scripts in `src/` to regenerate the derived data before launching the complete application.

## Research Limitations

- Historical survey data does not provide live hotel prices or current availability.
- Destination recommendations depend on the coverage of the curated attraction catalogue.
- Tourist clusters show overlap and should be interpreted cautiously.
- Expenditure estimates are model predictions, not guaranteed actual spending.

## Author

Machine Learning Capstone Project

## License

No license has been assigned yet. All rights reserved unless a license is added by the project owner.
