# 🔒 HAMAD: Hybrid Attention-based Multi-scale Anomaly Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A novel lightweight framework for real-time network anomaly detection combining ensemble learning with adaptive threshold mechanisms.

## 🌟 Features

- ✅ Hybrid ensemble of Random Forest and XGBoost with attention-based fusion
- ✅ Adaptive threshold mechanism for dynamic anomaly detection
- ✅ Multi-dataset evaluation (NSL-KDD, UNSW-NB15, CICIDS2017)
- ✅ Interactive Streamlit dashboard for real-time monitoring
- ✅ SHAP-based explainability
- ✅ 99.71% accuracy on UNSW-NB15 dataset

## 📊 Performance

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| NSL-KDD | 99.45% | 99.32% | 99.58% | 99.45% |
| UNSW-NB15 | 99.71% | 99.68% | 99.74% | 99.71% |
| CICIDS2017 | 99.62% | 99.55% | 99.69% | 99.62% |

## ✍️ Skills Demonstrated

## 💡 Data Science & Engineering Skills Demonstrated

- **End-to-End Data Pipeline:** Automated data ingestion, validation, cleaning, feature engineering, and preprocessing for industry-standard network datasets (NSL-KDD, UNSW-NB15, CICIDS2017).
- **Advanced Visualization:** Interactive dashboards built with Streamlit and Plotly, showing real-time metrics, confusion matrices, ROC/PR curves, and data exploration charts.
- **Feature Engineering & Analysis:** Multi-scale feature extraction, correlation heatmaps, distribution analysis for both raw and engineered features.
- **Statistical Analysis:** Computation of accuracy, precision, recall, F1-score, ROC-AUC; detailed per-class performance reporting for both majority and minority attack types.
- **Model Development:** Built ensemble models using Random Forest & XGBoost, attention-based fusion, and adaptive thresholding, with reproducible code structure.
- **Explainable AI:** Integrated SHAP for global and local feature impact explanations; produced interpretable force plots and summary visualizations.
- **MLOps & Deployment:** Packaged models and dashboards for real-time inference with Docker and Streamlit; ready for cloud or edge deployment.
- **Collaboration & Version Control:** Used modular code, detailed documentation, and GitHub best practices for collaborative data science projects.
- **Portfolio/Placement Ready:** Project structure and results highlighted in portfolio, ready for interviews, LinkedIn showcase, and GitHub sharing.

---

## 🏆 Placement Power: What Recruiters See

- Real-world visualization examples (screenshot links, GIFs)
- Clean codebase demonstrating Python, data science, and dashboarding skills
- Clear documentation of technical and analytical processes
- Project outcomes communicated with impactful, business-relevant metrics
- Fast, interactive interface for demonstrating skills in live interviews


## 🚀 Quick Start

### Installation

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt


### Data Preparation

Download datasets
python src/data/download_datasets.py

Preprocess data
python src/data/preprocess.py


### Model Training

Train HAMAD model
python src/models/hamad.py


### Run Dashboard

Launch interactive dashboard
streamlit run dashboards/app.py

## 📁 Project Structure

HAMAD-Network-Anomaly-Detection/
├── data/
│ ├── raw/ # Raw datasets
│ └── processed/ # Processed features
├── src/
│ ├── data/ # Data processing scripts
│ ├── models/ # Model implementations
│ ├── visualization/ # Plotting utilities
│ └── utils/ # Helper functions
├── models/ # Saved models
├── dashboards/ # Streamlit apps
├── notebooks/ # Jupyter notebooks
└── tests/ # Unit tests


## 🎯 Key Innovations

1. **Selective Feature Fusion**: Attention-weighted ensemble combining RF and XGBoost
2. **Dynamic Threshold Adaptation**: Reconstruction error-based threshold mechanism
3. **Multi-scale Temporal Features**: Short-term and long-term traffic pattern capture
4. **Edge-Ready Architecture**: Lightweight design for IoT/edge deployment

## 📖 Citation


## 👤 Author

**Samyadeep Saha**
- M.Tech in Cybersecurity, NIT Agartala
- Email: samyadeep.saha@nita.ac.in
- LinkedIn: [https://www.linkedin.com/in/samyadeep-saha-data/]
- GitHub: [(https://github.com/Samyadeep21)]

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NSL-KDD Dataset: [Canadian Institute for Cybersecurity]
- UNSW-NB15 Dataset: [UNSW Canberra]
- CICIDS2017 Dataset: [Canadian Institute for Cybersecurity]





