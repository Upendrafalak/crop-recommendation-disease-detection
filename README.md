# Farming Assistant

## Demo Video


https://user-images.githubusercontent.com/92196450/232058436-5593ed9b-b11e-49c4-afc0-6017568b4e80.mp4

## Problem Statement

1. Agriculture is vital to the global economy and supports more than 70% of rural families. Over the last few decades, agriculture has grown at a rapid pace.
2. The problems that the farmers face are poor health of the crops which results in unpredictable yield, and inconsistency in crop profits. 
3. Thus, there is a need to develop an impactful solution by predicting the crop disease by analyzing the photograph and giving suggestions to the farmers about its nurturing and various ways to maximize yield. 
4. Also, we need to recommend proper crops and fertilizers and from where they can get them and how they can use them.

## Idea
1. Crop images will be used as input for the ML model
2. ML model predicts disease, reasons for occurrence, and prevention/cure steps
3. Environmental factors such as temperature, wind, humidity, soil type, and nutrients will be used to predict suitable fertilizers
4. Suggestions for crops and fertilizers will be given to farmers to maximize profits
5. A web interface with an internet connection will be used
6. Information will be provided in the native language of farmers in different regions of India.

## User Diagram
![user_diagram](https://github.com/Upendrafalak/crop-recommendation-disease-detection/assets/92196450/6b0987e1-12b9-4fd5-a16d-16cffc92956c)

## Features
1. Plant Disease Detection
2. Crop Recommendation
3. Fertilizer Recommendation
4. Price Prediction
5. Weather Prediction
6. Fertilizer Shops & Soil Testing Labs Locator
7. E-Marketplace
8. Agricultural News 
9. Multilingual Support
10. Chatbot

## Tech Stack
1. ML Libraries: TensorFlow, Scikit-learn, Numpy, Pytorch, Pandas, Seaborn, Theano, Pickle
2. Frontend: NextJS, Tailwind CSS
3. Backend: Flask
4. Database: MongoDB
5. External Tools: Jupyter notebook, VS code, Kaggle

## Dataset
1. [Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

## Use Cases
1. Disease detection can help farmers use preventive measures and save time while monitoring crop health.
2. Fertilizer forecasting can help replenish soil nutrients and increase crop yield while reducing the likelihood of nutrient deficiency and land infertility.
3. Crop recommendations can assist farmers in selecting the best crops to grow based on environmental factors and maximize profits.
4. Accurate weather data can assist farmers in determining the best time to cultivate crops and prevent the spread of pests and crop diseases.


## Challenges we ran into

1. Increasing accuracy of model: By tweaking with hyperparameters and trying different models to increase the accuracy.
2. Isolated models: Integrated different models which will give each other inputs.
3. Accessing website might be difficult for farmers: Created a PWA solution for farmers.
