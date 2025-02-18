# Weather-prediction-ML

Weather Prediction Analysis
This repository contains a comprehensive analysis of weather data from Kanpur, utilizing various regression models to predict temperature. The dataset, kanpur (1).csv, includes features such as maximum and minimum temperatures, humidity, cloud cover, heat index, pressure, and wind speed, recorded over a specified time period.

Key Steps in the Analysis:
Data Preprocessing: The dataset is loaded, and relevant numerical features are extracted. Missing values are handled, and the data is visualized through various plots to understand distributions and trends.

Exploratory Data Analysis (EDA): Scatter plots are created to visualize relationships between temperature and other features, such as minimum temperature, heat index, and pressure.

Model Training and Evaluation:

Linear Regression: A linear regression model is trained and evaluated, with metrics such as mean absolute error and R² score calculated to assess performance.
Decision Tree Regressor: A decision tree model is also trained, and its performance is compared to the linear regression model.
Random Forest Regressor: Finally, a random forest model is implemented, providing a robust prediction capability.
Performance Metrics: The models' performances are evaluated using mean absolute error, residual sum of squares, and R² scores, allowing for a comprehensive comparison of their predictive capabilities.

Correlation Analysis: A correlation matrix is generated to identify relationships between different features, visualized using a heatmap for better interpretability.

Conclusion
This analysis demonstrates the application of machine learning techniques to weather prediction, highlighting the effectiveness of different regression models. The findings can be useful for further research in meteorology and climate science, as well as for practical applications in weather forecasting.

Requirements
To run the analysis, ensure you have the following Python libraries installed:

numpy
pandas
matplotlib
seaborn
scikit-learn
Usage
Clone the repository and run the Jupyter Notebook to reproduce the analysis. Adjust the dataset path as necessary to point to your local copy of the kanpur (1).csv file.
