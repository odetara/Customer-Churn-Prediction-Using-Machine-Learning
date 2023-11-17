# Customer-Churn-Prediction-Using-Machine-Learning

![image](https://www.neenopal.com/images/Understanding%20Customer%20Churn.png)

**Introduction:**

In the dynamic landscape of telecommunications, customer churn poses a significant challenge for companies like ConnectTel. The ability to predict and understand customer churn is crucial for business sustainability and growth. In this project, I delve into the realm of churn prediction, leveraging machine learning techniques to develop models capable of identifying customers at risk of leaving the service.

**Project Background**

ConnectTel is facing a client retention difficulty that threatens the company's long-term viability and growth. Customer churn prediction predicts potential customers to leave a company's service, requiring effective marketing strategies to increase their likelihood of staying.

**Project Objective**

The primary goal is to develop an accurate and reliable predictive model using machine learning to predict which customers are likely to churn and implement proactive measures.

**Data Loading and Cleaning**

The project commenced with the crucial phase of data loading and cleaning, which involved:
•	Preview the dataset to familiarize myself with its structure and contents.
•	Identified and standardized data types for consistency.
•	Detected and eliminated duplicate entries to ensure data integrity.


**Exploratory Data Analysis (EDA)**

To uncover trends or patterns, obtain insights, and remove unnecessary values from the data, I conducted univariate, bivariate, and multivariate analyses to gain an in-depth knowledge of the data and learn about its various features.

**Data Preprocessing:**

**1.	Feature Engineering:**

•	Identified and created relevant features that can contribute to the prediction of churn.
•	Handled missing values and outliers appropriately.

**2.	Encoding:**
  	
•	I used label encoding to convert categorical variables into a format suitable for machine learning models.

**3.	Scaling:**
  	
•	I normalized numerical features to ensure a level playing field for machine learning algorithms.

**Model Building:** Split the dataset into a training set and a testing set. A common split is 80% for training and 20% for testing.
•	Dataset Split: The dataset was divided into 80% for training and 20% for testing.
•	Models Implemented:

	Logistic Regression
	Random Forest
	Decision Tree

**Model Evaluation:**

The performance of each model was evaluated using key metrics:
•	Accuracy: overall correctness of the model predictions.
•	Precision: proportion of true positives among instances predicted as positive.
•	Recall: proportion of true positives among actual positive instances.
•	AUC (Area Under the Curve): the area under the ROC curve, measuring the model's ability to distinguish between classes.

**Key Insights:**

•	Logistic Regression stands out with the highest accuracy and a balanced precision-recall trade-off.
•	Decision Tree shows lower precision and recall compared to other models.
•	Random Forest provides a decent accuracy but has lower precision and recall compared to Logistic Regression. It may benefit from tuning hyperparameters to improve performance.
•	Naïve Bayes is notable for high recall, making it suitable for scenarios where capturing all churn instances is crucial.
•	Consider business priorities and the cost associated with false positives and false negatives when choosing a model.
Confusion Matrix Analysis

**Interpretation:**

•	True Positives (TP): The number of customers correctly predicted as churners.
•	True Negatives (TN): The number of customers correctly predicted as non-churners.
•	False Positives (FP): The number of customers incorrectly predicted as churners.
•	False Negatives (FN): The number of customers incorrectly predicted as non-churners.

**Model Comparison:**

•	Logistic Regression has the highest True Positives (214) and the fewest False Positives (159), indicating better performance in identifying actual churners.
•	Random Forest has slightly fewer True Positives but also fewer False Positives compared to Decision Tree.
•	Decision Tree has the highest False Positives and False Negatives, suggesting lower precision and recall compared to the other models.

**Recommendation:**

•	Logistic Regression appears to be performing better in this scenario, but the choice of the model depends on the specific goals and requirements of the business. Consider the trade-off between false positives and false negatives based on the business impact of predicting churn incorrectly.

The Logistic Regression model has a balanced distribution of false positives and false negatives. It demonstrates a higher ability to correctly identify non-churn instances (True Negatives) compared to correctly identifying churn instances (True Positives). The model's precision and recall can be further optimized by adjusting the classification threshold.

Similar to Logistic Regression, Random Forest exhibits a balanced distribution of false positives and false negatives. It performs slightly worse in correctly identifying both churn and non-churn instances compared to Logistic Regression. Random Forest's strength lies in ensemble learning, providing robustness against overfitting.

The Decision Tree model demonstrates a higher rate of false positives compared to both Logistic Regression and Random Forest. It shows comparable performance in correctly identifying churn instances (True Positives) but struggles with precision due to a higher false positive count. Decision Trees may benefit from pruning or tuning hyperparameters to improve overall performance.

**General Observations:**

The models generally perform better at identifying non-churn instances (True Negatives) than identifying churn instances (True Positives).
The choice between models depends on the specific business requirements and the importance of precision and recall in the context of customer churn.

**Conclusion and Recommendations:**

In conclusion, the project highlights the potential of machine learning in predicting customer churn. ConnectTel can benefit from deploying the Logistic Regression model for its superior performance. Explore ensemble methods or model stacking to combine the strengths of different models.

Recommendations include continuous monitoring, periodic model updates, and leveraging insights from misclassifications to enhance model robustness and business strategies. Fine-tune model parameters and thresholds to balance precision and recall based on business goals. By prioritizing precision and recall, ConnectTel can strategically address customer churn, fostering long-term customer relationships and business success.
