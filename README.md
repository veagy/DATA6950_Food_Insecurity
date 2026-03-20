# DATA6950_Food_Insecurity
# Predicting Food Insecurity Across U.S. Census Tracts  
### A Machine Learning Analysis Using the USDA Food Access Research Atlas

## Abstract
Food insecurity remains a persistent public health and socioeconomic challenge in the United States, disproportionately affecting low-income and geographically isolated communities. This study applies machine learning techniques to predict food access vulnerability across U.S. census tracts using data from the USDA Food Access Research Atlas (2010, 2015, and 2019 releases). We integrate demographic, economic, and geographic indicators including low-income and low-access population shares, poverty rate, SNAP participation, income levels, housing characteristics, and urban status to model food insecurity risk.
Three classification models were evaluated: Logistic Regression, Random Forest, and Extreme Gradient Boosting (XGBoost). Model performance was assessed using cross-validated ROC-AUC, accuracy, precision, recall, and F1-score metrics to account for class imbalance. Ensemble tree-based methods outperformed Logistic Regression, with XGBoost achieving the highest predictive performance (ROC-AUC ≈ 0.945; accuracy ≈ 0.92), followed closely by Random Forest. Feature importance analysis revealed that low-income and low-access population shares, household income, poverty rate, and SNAP participation were the most influential predictors.
The findings demonstrate that nonlinear machine learning models capture complex socioeconomic and spatial interactions underlying food access vulnerability more effectively than traditional linear approaches. This study contributes a scalable, data-driven framework for identifying high-risk communities and informing targeted food policy interventions.

## Keywords
Food insecurity, Food access, Machine learning, XGBoost, Random Forest, USDA Food Access Research Atlas, Socioeconomic determinants, Public policy analytics, Spatial inequality, Predictive modeling.

## 1. Introduction
Food insecurity remains a persistent and complex public health challenge in the United States. It affects millions of households and is driven by interconnected structural factors such as poverty, unemployment, geographic isolation, and limited access to affordable and nutritious food (Coleman-Jensen et al., 2023). According to the U.S. Department of Agriculture (USDA), approximately 13.5% of U.S. households nearly 18 million experienced food insecurity at some point in 2023 (U.S. Department of Agriculture [USDA], 2024), representing one of the highest levels observed in the past decade. These disparities disproportionately affect low-income populations, racial and ethnic minorities, and residents of rural communities, reflecting broader patterns of socioeconomic inequality (Gundersen & Ziliak, 2021).
The concept of "food deserts" serves as an established framework that researchers use to analyze geographic differences in food access. Food retailers need to maintain spatial proximity to their locations because this factor affects consumer access (Walker et al., 2010). Prior research indicates that food insecurity extends beyond geographic boundaries because spatial proximity to food retailers constitutes one of its determining factors. Economic instability and transportation problems and residential segregation and unemployment create major barriers that stop people from getting their food needs met (Bitto et al., 2003; Beaulac et al., 2009). The existing mapping and descriptive analysis methods serve their purpose of identifying present-day problems but they lack the ability to reveal the intricate relationships that exist between demographic, economic and environmental factors which determine food insecurity risk (Jiao et al., 2021).
In recent years, data-driven approaches have emerged as powerful tools for analyzing and forecasting social and health disparities. Machine learning techniques, in particular, offer advantages over conventional statistical models by identifying nonlinear relationships and high-order interactions among predictors. In addition to improving predictive performance, modern interpretability techniques such as SHAP (SHapley Additive exPlanations) values and partial dependence analysis allow researchers to better understand the contribution of individual variables to model outcomes. These advancements align with the growing national emphasis on evidence-based policymaking and targeted, data-informed community interventions.
This study aims to develop an interpretable machine learning framework to predict food insecurity across U.S. census tracts using the USDA Food Access Research Atlas datasets from 2010, 2015, and 2019 (U.S. Department of Agriculture, 2010, 2015, 2019). Specifically, the research compares the performance of Logistic Regression, Random Forest, and XGBoost classifiers in identifying high-risk areas classified as food deserts. Beyond evaluating predictive accuracy, the study investigates the socioeconomic and demographic factors most strongly associated with food insecurity risk. By leveraging nationally representative, open-access data, this research contributes to advancing predictive analytics for equitable food access and supports more targeted public health and urban planning strategies.

## 2. Literature Review
### 2.1 Use of the USDA Food Access Research Atlas in Food Desert Research
The USDA Food Access Research Atlas has become one of the most widely used national datasets for examining spatial and socioeconomic disparities in food access in the United States. Developed by the United States Department of Agriculture Economic Research Service, the Atlas provides census tract–level measures of low-income and low-access status based on standardized distance thresholds and poverty criteria.
Several peer-reviewed studies have used the Food Access Research Atlas to examine the structural determinants of food deserts. Ver Ploeg et al. (2012), in a foundational USDA report, introduced the methodology underlying low-income and low-access tract classification, establishing the empirical framework used in subsequent academic research. Their work formalized the use of 1-mile urban and 10-mile rural thresholds in combination with income criteria, which now serve as the national benchmark for food desert identification.
More recent empirical studies have relied directly on the Atlas for spatial and socioeconomic modeling. Kolak et al. (2018) used the USDA Food Access Research Atlas in combination with health data to examine neighborhood-level disparities in food access and chronic disease outcomes. Their findings highlighted the importance of tract-level poverty concentration and demographic composition in predicting low-access status.
Similarly, Karpyn et al. (2019) utilized USDA Food Access Research Atlas measures to evaluate food retail access disparities across metropolitan areas. Their study emphasized that low-income status and transportation barriers significantly mediate geographic access to healthy food retailers.
Singleton et al. (2020) further demonstrated the applicability of the USDA Atlas for longitudinal and spatial analyses, using tract-level Food Access Research Atlas data to examine urban–rural disparities and demographic inequalities over time.
Collectively, these studies establish the USDA Food Access Research Atlas as a validated and nationally standardized dataset for food desert research. Its tract-level granularity, consistent definitions across years, and integration of income and distance metrics make it particularly suitable for multivariate modeling approaches such as those employed in the present study.
### 2.2 Quantitative and Machine Learning Applications Using Food Access Atlas Data
While early studies using the USDA Food Access Research Atlas relied primarily on descriptive statistics and regression models, more recent scholarship has incorporated advanced quantitative techniques to model structural determinants of food access.
Kolak et al. (2018) applied spatial statistical modeling to USDA Food Access Research Atlas data to identify geographic clustering of vulnerability and chronic disease. Their work underscores the importance of incorporating demographic exposure variables and poverty interactions when modeling food desert classification.
Additionally, Allcott et al. (2019), though focusing more broadly on nutritional inequality, used nationally representative food access data aligned with USDA definitions to demonstrate that income constraints often outweigh pure geographic proximity in explaining disparities in healthy food consumption. Their findings support the inclusion of poverty measures, SNAP participation, and socioeconomic vulnerability variables in predictive modeling.
Recent methodological work in public health analytics increasingly employs ensemble machine learning techniques, such as Random Forest and gradient boosting, to capture nonlinear and interactive effects in social determinants data (Breiman, 2001; Chen & Guestrin, 2016). These approaches are particularly appropriate for food access research, where poverty, transportation barriers, and demographic composition interact in complex ways.
Furthermore, the use of interpretability frameworks such as SHAP (Lundberg & Lee, 2017) aligns with emerging best practices in policy-oriented machine learning research, enabling transparent identification of the most influential tract-level predictors of food desert classification.
Thus, the integration of USDA Food Access Research Atlas data with ensemble machine learning models reflects both established data usage practices and contemporary analytical advancements in socioeconomic and public health research.
### 2.3 Conceptualizing Food Insecurity and Geographic Disparities
Food insecurity refers to limited or uncertain access to adequate food due to economic or social constraints. Research in public health and nutrition has documented its persistence in the United States, with millions of households experiencing disruptions in food availability and access (Coleman-Jensen et al., 2023). Food insecurity exists as a social economic problem because it arises from poverty and inequality, yet studies about spatial distribution patterns show that different communities have different food resource availability. Researchers developed the definition of “food deserts” to describe places where people cannot reach supermarkets or obtain affordable healthy food (Walker et al., 2010). Researchers believe that food desert definition needs to extend beyond geographic boundaries because income level and transportation system and societal structures affect food access (Beaulac et al., 2009; Bitto et al., 2003). The research demonstrates that food insecurity problems need analytical methods which can assess both spatial factors and economic conditions.
### 2.4 Traditional Approaches to Food Access Measurement
Conventional studies of food access have primarily employed descriptive mapping and regression-based analyses to identify areas with limited food resources. Geographic Information Systems (GIS) have been used to locate food deserts and assess proximity to supermarkets, often revealing clusters of low-income and minority neighborhoods with poor access (Walker et al., 2010). The spatial analyses used in local policy interventions have produced useful results, but their capacity to assess multiple risk factors remains restricted. Beaulac et al. (2009) conducted a systematic review of food desert research, which showed that researchers follow inconsistent methods to define and measure food access because most studies only present permanent descriptions that do not predict future developments. Bitto et al. (2003) discovered that rural communities face food access problems which extend beyond their distance to grocery stores because transportation problems and financial instability create additional obstacles. The traditional methods of food access assessment can create descriptions of food distribution patterns but they fail to capture the intricate system of factors that lead to food security problems in various situations.
### 2.5 Machine Learning in Food Insecurity and Health Disparity Prediction
Recent advancements in data science have introduced machine learning techniques as powerful alternatives for modeling complex social phenomena, including food insecurity. Machine learning algorithms handle high-dimensional data because they can identify nonlinear relationships. This capability makes machine learning algorithms suitable for prediction tasks (Jiao et al., 2021). Machine learning has helped food access researchers locate high-risk areas through demographic, economic, and environmental data integration. The research by Jiao et al. (2021) shows that using spatial and socioeconomic data with advanced models leads to better food access assessment results than traditional GIS mapping methods. Policymakers can use interpretable machine learning tools such as SHAP (SHapley Additive exPlanations) and partial dependence plots to see how each variable affects model results (Molnar, 2020). Machine learning methods use nationwide representative data from the USDA Food Access Research Atlas to improve scientific research and practical application for measuring food security and planning interventions.

## 3. Data and Methodology
### 3.1 Dataset Overview
This study utilizes data from the U.S. Department of Agriculture Economic Research Service (USDA ERS) Food Access Research Atlas. The Food Access Research Atlas is a nationally representative, census-tract-level dataset designed to measure geographic and socioeconomic barriers to food access across the United States. It integrates demographic, income, and spatial accessibility indicators to identify communities that may be at risk of limited access to affordable and nutritious food.
The analysis used three data releases from the years 2010, 2015 and 2019. The selected years enable researchers to study how food access conditions changed over almost ten years while using the same variable definitions and census-tract boundaries for their research. The study gains more participants through multiple study years which creates better predictive models that can handle various socioeconomic and geographic factors. The study investigates U.S. census tracts as its main research unit. Census tracts function as permanent statistical divisions of counties which show consistent demographic and economic patterns of their residents. The system enables researchers to find food access differences in specific areas which helps them create effective solutions.
The primary outcome variable was derived from the USDA indicator LILATracts_1And10, which identifies census tracts classified as both low income and low access. According to USDA definitions, a tract is considered low income if it meets poverty or median family income thresholds, and low access if a significant portion of the population resides beyond one mile (urban areas) or ten miles (rural areas) from the nearest supermarket or large grocery store. This variable was converted into a binary classification target, where:
•	1 = Food desert (low income & low access)
•	0 = Non–food desert
Predictor variables extracted from the dataset include socioeconomic indicators (poverty rate, median family income), demographic composition (race/ethnicity proportions, youth and senior population shares), transportation access (households with no vehicle), urban/rural classification, and SNAP participation measures. These variables were selected based on prior literature identifying economic vulnerability, demographic structure, and transportation barriers as key determinants of food insecurity risk.
By leveraging a nationally standardized dataset spanning multiple years, this study provides a consistent and scalable framework for predictive modeling of food desert designation. The use of publicly available USDA data enhances transparency, reproducibility, and policy relevance, making the findings applicable to federal, state, and local decision-makers seeking data-informed strategies to address food access inequities.
### 3.2 Data Preprocessing and Feature Engineering
The raw datasets from 2010, 2015, and 2019 were first merged into a single consolidated dataframe. A new variable, year, was appended to each dataset prior to concatenation to preserve temporal information. Merging multiple releases increased the total number of observations and allowed the model to capture variation in food access conditions across time.
Data Type Conversion
The datasets were imported with string data types to preserve formatting consistency across Excel and CSV files, relevant predictor variables were explicitly converted to numeric format. The selected numeric features included poverty rate, median family income, SNAP participation share, Racial exposure within low-access, Vulnerable population exposure. Conversion errors were coerced into missing values to ensure data integrity and prevent model distortion.
The target variable, derived from the USDA LILATracts_1And10 indicator, was also converted to numeric format and treated as a binary classification label.
Missing Data Handling
Missing values were assessed using feature-level missingness proportions. A bar plot visualization was generated to evaluate the percentage of missing data across predictors.
Given the relatively moderate and non-systematic nature of missing values, median imputation was applied using SimpleImputer. Median imputation was selected because it is robust to skewed distributions (particularly income-related variables) and reduces sensitivity to extreme outliers.
Observations with missing target labels were removed to ensure model validity.
Exploratory Data Analysis (EDA)
Exploratory analysis was conducted to understand the structure and distribution of key variables prior to modeling. This included:
•	Missing Data Assessment: The proportion of missing values across predictor variables is illustrated in (Fig1). Overall, missingness appears limited and concentrated within a small subset of features. Most variables demonstrate relatively low proportions of missing observations, suggesting that the dataset is largely complete and suitable for modeling without extensive exclusion of records. Given the distribution of missingness, median imputation was applied to numeric predictors. This approach is appropriate for socioeconomic variables such as income and poverty rate, which often exhibit skewed distributions. The limited level of missing data supports the reliability and stability of the subsequent machine learning analysis

•	Class distribution visualization(Fig 2) to assess imbalance between food desert and non–food desert tracts. While the majority of tracts fall into the non–food desert category, a substantial minority are classified as food deserts. This imbalance justifies the use of stratified sampling during train-test splitting and cross-validation, as well as the application of class-weight adjustments in Logistic Regression and Random Forest models. Proper handling of class imbalance is essential to ensure that predictive performance is not artificially inflated by the majority class.
•	Histograms of poverty rate, median income, and SNAP participation: The distribution of key predictors (Fig 3) shows clear socioeconomic and geographic inequality across U.S. census tracts. Income levels vary widely, with many tracts clustered at lower income levels, while poverty rates are right-skewed, indicating that a smaller group of communities experience concentrated economic hardship. These economically disadvantaged tracts are more likely to be classified as food insecure. Similarly, low-income and low-access population shares are unevenly distributed, with certain areas facing much higher levels of combined economic and geographic barriers. SNAP participation follows a comparable pattern, reflecting concentrated need in vulnerable communities. Overall, these skewed and clustered distributions suggest that food insecurity is not evenly spread but concentrated in structurally disadvantaged areas. This pattern supports the use of machine learning models, which better capture nonlinear relationships and interactions among these predictors.
 
•	Correlation heatmap: To better understand the structural relationships among predictor variables, a correlation heatmap was constructed(Fig 4). The heatmap reveals patterns of association between income, poverty, vehicle access, and demographic indicators. The analysis shows a strong negative correlation between income-related variables and poverty indicators. This inverse relationship aligns with established socioeconomic theory, where higher income levels are associated with lower poverty rates. Additionally, variables representing poverty and lack of vehicle access demonstrate positive correlation, suggesting that transportation vulnerability frequently coexists with economic disadvantage. Importantly, no extreme correlations exceeding conventional multicollinearity thresholds were observed. This indicates that while variables are related in theoretically coherent ways, they do not exhibit redundancy severe enough to destabilize model estimation. The absence of excessive multicollinearity strengthens confidence in both the interpretability of Logistic Regression coefficients and the stability of ensemble-based model predictions.
 
Feature Engineering
To enhance predictive performance and capture nonlinear socioeconomic effects, several derived variables were constructed:
1.	Log-Transformed Median Income (log_income)
Median family income was log-transformed using the natural logarithm transformation:
                     log(1+MedianFamilyIncome) 
This transformation reduces skewness and stabilizes variance, improving model interpretability and performance.
2.	Poverty–No Vehicle Interaction (poverty_no_vehicle)
An interaction term between poverty rate and vehicle access indicator was constructed to capture compounded transportation disadvantage:
PovertyRate×HUNVFlag 
This feature reflects the combined effect of economic vulnerability and limited mobility.
These engineered features allow the model to detect structural vulnerabilities that may not be captured by individual predictors alone.
Feature Selection
Final predictor variables included:
•	Poverty rate
•	Log-transformed income
•	SNAP participation share
•	Vehicle access indicator
•	Urban classification
•	Senior population
•	Racial and ethnic composition variables
•	Poverty–vehicle interaction term
This feature set was informed by both domain literature and exploratory correlation analysis.
Train-Test Split
The dataset was partitioned into training (80%) and testing (20%) subsets using stratified sampling to preserve class proportions. Additionally, 5-fold stratified cross-validation was employed during model evaluation to ensure robust and unbiased performance estimation.
### 3.3 Machine Learning Models
To predict food insecurity across U.S. census tracts, three supervised classification algorithms were implemented: Logistic Regression, Random Forest, and XGBoost. These models were selected to balance interpretability and predictive performance, as well as to account for potential nonlinear relationships among socioeconomic and demographic predictors.
## Logistic Regression

Logistic Regression is used as a baseline model due to its simplicity and interpretability. It estimates the probability that a census tract is classified as a food desert using a linear combination of input features passed through a sigmoid (S-shaped) function.

**Equation:**

P(y = 1 | X) = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₚxₚ))

Where:
- P(y = 1 | X): Probability of being a food desert
- x₁, x₂, ..., xₚ: Input features
- β₀, β₁, ..., βₚ: Model coefficients

The model outputs values between 0 and 1. A threshold (typically 0.5) is applied:
- If probability ≥ 0.5 → Food Desert (1)
- If probability < 0.5 → Not a Food Desert (0)

**Model Improvements:**
- Feature standardization to ensure consistent scale
- Class weighting to handle imbalanced data

## Random Forest

Random Forest is an ensemble learning method that builds multiple decision trees using different subsets of the data and features. The final prediction is obtained by combining the outputs of all trees.

**Equation:**

ŷ = mode(T₁(X), T₂(X), ..., Tₙ(X))

Where:
- Tᵢ(X): Prediction from the i-th decision tree
- ŷ: Final predicted class (majority vote)

Each tree makes its own prediction, and the final result is determined by majority voting.

**Model Configuration:**
- 300 decision trees
- Minimum leaf size: 5 samples
- Balanced class weights
- Parallel processing enabled

**Advantages:**
- Captures nonlinear relationships
- Handles feature interactions
- Provides feature importance scores

## XGBoost (Gradient Boosting)

XGBoost is a gradient boosting algorithm that builds decision trees sequentially. Each new tree focuses on correcting the errors made by previous trees.

**Objective Function:**

L = Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)

Where:
- l(yᵢ, ŷᵢ): Loss function (prediction error)
- Ω(fₖ): Regularization term to control model complexity

Unlike Random Forest, XGBoost builds trees one after another, improving performance at each step.

**Key Hyperparameters:**
- 300 estimators
- Maximum depth: 5
- Learning rate: 0.05
- Subsampling of rows and features

**Advantages:**
- High predictive accuracy
- Handles complex patterns
- Reduces overfitting through regularization

All models were trained on the stratified training set, and performance was evaluated using 5-fold cross-validation with ROC-AUC as the primary metric. Final evaluation metrics, including precision, recall, F1-score, confusion matrices, and ROC curves, were computed on the held-out test set to assess generalization.
### 3.4 Model Evaluation and Interpretability
Model performance was evaluated using multiple metrics to provide a comprehensive assessment of predictive accuracy and reliability. The primary metric was the Receiver Operating Characteristic – Area Under the Curve (ROC-AUC), which measures the model’s ability to distinguish between food desert and non–food desert tracts. ROC-AUC was chosen because it is robust to class imbalance and provides a threshold-independent evaluation. Additional metrics included precision, recall, F1-score, and confusion matrices, which offer insight into the models ability to correctly identify both positive and negative instances.
Stratified 5-fold cross-validation was employed to estimate out-of-sample performance and mitigate overfitting. This procedure ensures that the proportion of food desert and non–food desert tracts is preserved in each fold, resulting in more reliable and unbiased estimates of model performance.
The analysis needs interpretability because it delivers crucial insights that policymakers can use to make decisions. The Random Forest and XGBoost models used feature importance assessment to measure how each predictor affected the models' decision-making process. The Random Forest model used SHapley Additive exPlanations (SHAP) values to establish a reliable method for measuring how different features affected the model's performance. The SHAP summary plots together with the bar charts show how each feature affects predicted probabilities which enables a clear view of how socioeconomic and demographic elements determine food desert classification. The combination of performance metrics and interpretability techniques ensures that the predictive models are not only accurate but also informative. The study identifies key predictors which enables the study to calculate their influence thus providing evidence-based insights which will help to develop effective interventions for high-risk communities experiencing food insecurity.

## 4. Results
This section presents the comparative performance of Logistic Regression, Random Forest, and XGBoost models in predicting food insecurity across U.S. census tracts using the USDA Food Access Research Atlas data (2010, 2015, 2019). Model performance was evaluated using accuracy, precision, recall, F1-score, and cross-validated ROC-AUC. Due to class imbalance, particular attention is given to minority-class performance (food insecure tracts).
### 4.1 Model Performance Comparison
Table 1 presents the classification results for the three models. Logistic Regression achieved an overall accuracy of 0.82 and a cross-validated ROC-AUC of 0.907. While recall for the minority class (food insecure tracts) was relatively high (0.82), precision was low (0.43), indicating a high number of false positives. This suggests that Logistic Regression tends to over-predict food insecurity.
Random Forest substantially improved overall performance, achieving 0.90 accuracy and a ROC-AUC of 0.942. The model produced a more balanced tradeoff between precision (0.62) and recall (0.74) for the minority class, resulting in a higher F1-score (0.67). This indicates improved reliability in identifying vulnerable census tracts while reducing false classifications.
XGBoost achieved the highest overall accuracy (0.92) and the strongest discriminative ability, with a ROC-AUC of 0.945. The model demonstrated higher precision for the minority class (0.74) but lower recall (0.61), indicating a more conservative prediction strategy. While it reduced false positives, it missed a larger portion of truly vulnerable tracts compared to Random Forest.
Overall, ensemble tree-based methods outperformed Logistic Regression, highlighting the importance of capturing nonlinear interactions among socioeconomic and geographic predictors.
### Table 1: Classification Performance Comparison

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) | ROC-AUC |
|------|---------|---------------------|------------------|--------------------|--------|
| Logistic Regression | 0.82 | 0.43 | 0.82 | 0.56 | 0.907 |
| Random Forest | 0.90 | 0.62 | 0.74 | 0.67 | 0.942 |
| XGBoost | 0.92 | 0.74 | 0.61 | 0.67 | 0.945 |

### 4.2 ROC Curve Analysis
Figure 5 displays the ROC curves for all three models. All models performed substantially above the random classification line, confirming strong predictive capability. Logistic Regression demonstrated the lowest area under the curve, while Random Forest and XGBoost exhibited superior and closely aligned performance.
The marginal improvement of XGBoost over Random Forest suggests that gradient boosting slightly enhances discriminative capacity, though both ensemble approaches significantly outperform the linear baseline model.

### 4.3 Confusion Matrix Analysis
Figure 6 presents the confusion matrix for the XGBoost model, which achieved the highest overall predictive performance. The model correctly classified 11,215 non-vulnerable tracts (true negatives) and 1,120 vulnerable tracts (true positives). However, 722 vulnerable tracts were misclassified as non-vulnerable (false negatives), and 401 non-vulnerable tracts were incorrectly predicted as vulnerable (false positives).
This pattern reflects the model’s higher precision but comparatively lower recall for the minority class, indicating that it prioritizes reducing false alarms at the expense of missing some truly food-insecure areas.
 
### 4.4 Feature Importance Analysis
Feature importance derived from the Random Forest model is presented in Figure 7. The most influential predictors include low-income and low-access population share (lalowihalfshare), log-transformed income, SNAP participation share (lasnaphalfshare), poverty rate, and low-access population share (lapophalfshare). Housing vulnerability and urban classification also contributed meaningfully to prediction.
These findings confirm that both economic disadvantage and geographic food access barriers jointly shape food insecurity risk. The dominance of low-income and low-access measures underscores the structural nature of food insecurity across U.S. census tracts.
 
### 4.5 Summary of Findings
The results demonstrate that ensemble machine learning methods significantly outperform Logistic Regression in predicting food insecurity. While XGBoost achieved the highest overall accuracy and ROC-AUC, Random Forest provided a more balanced precision–recall tradeoff for identifying vulnerable tracts. Feature importance results highlight the central role of income, poverty, SNAP participation, and low-access measures in shaping food insecurity risk.
Together, these findings suggest that food insecurity is driven by complex, nonlinear interactions between socioeconomic and spatial determinants, reinforcing the value of advanced machine learning approaches for policy-relevant predictive modeling.

## 5. Discussion
This study examined the predictive capacity of machine learning models in identifying food insecurity risk across U.S. census tracts using the USDA Food Access Research Atlas data. The results demonstrate that ensemble tree-based methods, particularly Random Forest and XGBoost, substantially outperform Logistic Regression in predictive accuracy and discriminative power. These findings suggest that food insecurity is shaped by complex, nonlinear relationships among socioeconomic and geographic factors that are not fully captured by traditional linear models.
The superior performance of XGBoost and Random Forest indicates that interactions between poverty, income, SNAP participation, and geographic access measures play a critical role in determining vulnerability. Logistic Regression, while achieving strong recall for the minority class, produced a high rate of false positives, reflecting limitations in modeling threshold effects and nonlinear dependencies. In contrast, ensemble models better captured these structural patterns, resulting in higher ROC-AUC values and improved balance between precision and recall.
Feature importance analysis further reinforces the structural nature of food insecurity. Variables representing low-income and low-access population shares emerged as the strongest predictors, followed by poverty rate, median income, and SNAP participation. These findings align with existing literature that identifies economic deprivation and spatial barriers as central determinants of food access disparities. Importantly, the prominence of combined low-income and low-access indicators suggests that vulnerability arises not merely from poverty alone, but from the intersection of economic disadvantage and geographic isolation.
The distributional patterns observed in the predictors particularly their skewed and clustered nature highlight the uneven geographic concentration of food insecurity risk. A relatively small subset of census tracts bears a disproportionate burden of socioeconomic disadvantage and limited food access. This clustering effect underscores the importance of targeted, place-based policy interventions rather than broad, uniform strategies.
From a policy perspective, the findings demonstrate the value of machine learning as a screening tool for identifying high-risk communities. The ability of ensemble models to accurately classify vulnerable tracts provides policymakers with a scalable framework for prioritizing resource allocation, expanding SNAP outreach, and addressing structural food access barriers. However, the tradeoff between precision and recall observed in XGBoost suggests that model selection should align with policy objectives whether minimizing missed vulnerable communities or reducing false identification of non-vulnerable areas.
Several limitations should be acknowledged. First, the study relies on publicly available census-tract-level data, which may not capture household-level food insecurity experiences. Second, while the models demonstrate strong predictive performance, they do not establish causal relationships between predictors and food insecurity. Future research could incorporate longitudinal modeling, spatial econometric approaches, or additional environmental and retail density measures to further refine predictive accuracy.
Overall, this study contributes to the growing literature on data-driven public policy by demonstrating that advanced machine learning techniques offer meaningful improvements over traditional statistical models in predicting food insecurity. The results underscore that food insecurity is a structurally embedded, multidimensional phenomenon requiring targeted and analytically informed intervention strategies.

### 5.1 Policy Implications
The results demonstrate that machine learning models can serve as effective tools for identifying census tracts at high risk of food insecurity. By integrating economic and geographic indicators such as poverty, income, SNAP participation, and low-access measures, policymakers can better target resources to communities with the greatest need.
The findings also highlight that food insecurity stems from the intersection of economic disadvantage and spatial barriers. This suggests that policies should address both affordability and physical access to food through targeted SNAP expansion, transportation support, and incentives for food retailers in underserved areas. Model selection can further be aligned with policy goals, depending on whether minimizing missed vulnerable communities or reducing false identification is prioritized.
### 5.2 Future Research
Future research could incorporate longitudinal analysis to examine how food insecurity risk changes over time. Including additional spatial variables, such as food retailer density or transportation infrastructure, may further improve predictive performance.
Moreover, combining machine learning with causal inference methods could strengthen understanding of the mechanisms driving food insecurity. Expanding to more granular, household-level data and applying model explainability techniques would also enhance transparency and policy relevance.
## Conclusion
This study applied machine learning techniques to predict food insecurity across U.S. census tracts using data from the USDA Food Access Research Atlas (2010, 2015, and 2019). The findings demonstrate that ensemble models, particularly Random Forest and XGBoost, outperform Logistic Regression in capturing the complex socioeconomic and geographic patterns underlying food access vulnerability. Key predictors including low-income and low-access population shares, poverty rate, income, and SNAP participation highlight the structural and spatial nature of food insecurity in the United States.
The results underscore the value of advanced predictive analytics in public policy research. By leveraging nonlinear modeling approaches, policymakers can more accurately identify high-risk communities and design targeted interventions. Overall, this study contributes a scalable, data-driven framework for understanding and addressing food insecurity through evidence-based decision-making.

## References (APA 7th Edition)
Beaulac, J., Kristjansson, E., & Cummins, S. (2009). A systematic review of food deserts, 1966–2007. Preventing Chronic Disease, 6(3), A105. https://www.cdc.gov/pcd/issues/2009/jul/08_0163.htm
Bitto, E. A., Morton, L. W., Oakland, M. J., & Sand, M. (2003). Grocery store access patterns in rural food deserts. Journal of Extension, 41(6). https://archives.joe.org/joe/2003december/a3.php
Coleman-Jensen, A., Rabbitt, M. P., Gregory, C. A., & Singh, A. (2023). Household food security in the United States in 2022 (Economic Research Report No. 325). U.S. Department of Agriculture, Economic Research Service. https://www.ers.usda.gov/publications/pub-details/?pubid=106979
Gundersen, C., & Ziliak, J. P. (2021). Food insecurity and health outcomes. Health Affairs, 34(11), 1830–1839. https://doi.org/10.1377/hlthaff.2015.0645
Jiao, J., Moudon, A. V., Ulmer, J., Hurvitz, P. M., & Drewnowski, A. (2021). How to identify food deserts: Measuring physical and economic access to supermarkets in King County, Washington. American Journal of Public Health, 102(10), e32–e39. https://doi.org/10.2105/AJPH.2012.300675
U.S. Department of Agriculture, Economic Research Service. (2024). Food security status of U.S. households in 2023. https://www.ers.usda.gov/topics/food-nutrition-assistance/food-security-in-the-u-s/
U.S. Department of Agriculture, Economic Research Service. (2010). Food Access Research Atlas (2010 data release). https://www.ers.usda.gov/data-products/food-access-research-atlas/
U.S. Department of Agriculture, Economic Research Service. (2015). Food Access Research Atlas (2015 data release). https://www.ers.usda.gov/data-products/food-access-research-atlas/
U.S. Department of Agriculture, Economic Research Service. (2019). Food Access Research Atlas (2019 data release). https://www.ers.usda.gov/data-products/food-access-research-atlas/
Walker, R. E., Keane, C. R., & Burke, J. G. (2010). Disparities and access to healthy food in the United States: A review of food deserts literature. Health & Place, 16(5), 876–884. https://doi.org/10.1016/j.healthplace.2010.04.013
Beaulac, J., Kristjansson, E., & Cummins, S. (2009). A systematic review of food deserts, 1966–2007. Preventing Chronic Disease, 6(3), A105. https://www.cdc.gov/pcd/issues/2009/jul/08_0163.htm
Bitto, E. A., Morton, L. W., Oakland, M. J., & Sand, M. (2003). Grocery store access patterns in rural food deserts. Journal of Extension, 41(6). https://archives.joe.org/joe/2003december/a3.php
Coleman-Jensen, A., Rabbitt, M. P., Gregory, C. A., & Singh, A. (2023). Household food security in the United States in 2022 (Economic Research Report No. 325). U.S. Department of Agriculture, Economic Research Service. https://www.ers.usda.gov/publications/pub-details/?pubid=106979
Jiao, J., Moudon, A. V., Ulmer, J., Hurvitz, P. M., & Drewnowski, A. (2021). How to identify food deserts: Measuring physical and economic access to supermarkets in King County, Washington. American Journal of Public Health, 102(10), e32–e39. https://doi.org/10.2105/AJPH.2012.300675
Molnar, C. (2020). Interpretable machine learning: A guide for making black box models explainable. Leanpub.
Walker, R. E., Keane, C. R., & Burke, J. G. (2010). Disparities and access to healthy food in the United States: A review of food deserts literature. Health & Place, 16(5), 876–884. https://doi.org/10.1016/j.healthplace.2010.04.013


