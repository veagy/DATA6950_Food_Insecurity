# DATA6950_Food_Insecurity
# Predicting Food Insecurity Across U.S. Census Tracts  
### A Machine Learning Analysis Using the USDA Food Access Research Atlas

##Abstract
Food insecurity remains a persistent public health and socioeconomic challenge in the United States, disproportionately affecting low-income and geographically isolated communities. This study applies machine learning techniques to predict food access vulnerability across U.S. census tracts using data from the USDA Food Access Research Atlas (2010, 2015, and 2019 releases). We integrate demographic, economic, and geographic indicators including low-income and low-access population shares, poverty rate, SNAP participation, income levels, housing characteristics, and urban status to model food insecurity risk.
Three classification models were evaluated: Logistic Regression, Random Forest, and Extreme Gradient Boosting (XGBoost). Model performance was assessed using cross-validated ROC-AUC, accuracy, precision, recall, and F1-score metrics to account for class imbalance. Ensemble tree-based methods outperformed Logistic Regression, with XGBoost achieving the highest predictive performance (ROC-AUC ≈ 0.945; accuracy ≈ 0.92), followed closely by Random Forest. Feature importance analysis revealed that low-income and low-access population shares, household income, poverty rate, and SNAP participation were the most influential predictors.
The findings demonstrate that nonlinear machine learning models capture complex socioeconomic and spatial interactions underlying food access vulnerability more effectively than traditional linear approaches. This study contributes a scalable, data-driven framework for identifying high-risk communities and informing targeted food policy interventions.

##Keywords
Food insecurity, Food access, Machine learning, XGBoost, Random Forest, USDA Food Access Research Atlas, Socioeconomic determinants, Public policy analytics, Spatial inequality, Predictive modeling.
