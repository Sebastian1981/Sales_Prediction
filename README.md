# Predict Bigmart´s Future Sales and Find Driving Features 

## Project Purpose
Predict sales for different Bigmart outlets using different machine learning regression models such as linear, tree and neural network models. Besides the sales forecast, we extract the feature driving sales by using a game-theoretical approach to make decision making of performant blackbox models transparent.
## Setup the Environment using Conda to run the JupyterNotebooks
- $conda create -n myenv python=3.8.11
- $conda active myenv
- $pip install -r requirements.txt

## Modeling Results & Evaluation
The figures 1-3 show the overall model performances for training data (blue) and test data (orange) using a simple linear model, a gradient-boosted tree model and a deep neural network model. Apparently, there is a good match beetween the predictions and the actual sales. The performance metric R2 for the linear model was r2=0.55 (training set) and r2=0.57 (test set). R2 for the gradient-boosted tree model was r2=0.61 (training & test set). R2 for the deep neural network model was r2=0.59 (training & test set). 

<table>
  <tr><td>
    <img 
        src="images/results_1.png"
        alt="Fashion MNIST sprite"  width="800">
  </td></tr>
    <tr><td align="center">
    <b>Figure 1. Comparison of true sales and model forecasts using a simple linear model. 
  </td></tr>
</table>

<table>
  <tr><td>
    <img 
        src="images/results_2.png"
        alt="Fashion MNIST sprite"  width="800">
  </td></tr>
    <tr><td align="center">
    <b>Figure 2. Comparison of true sales and model forecasts using a gradient-boosted tree model. 
  </td></tr>
</table>

<table>
  <tr><td>
    <img 
        src="images/results_3.png"
        alt="Fashion MNIST sprite"  width="800">
  </td></tr>
    <tr><td align="center">
    <b>Figure 3. Comparison of true sales and model forecasts using a deep neural network model with 2 hidden layers. 
  </td></tr>
</table>

## Model Interpretability Using Shapley Values from Game Theory
As described above, the neural-prophet model is highly interpretable due to its component-wise additive nature. The figure below show the different model components and their contribution to the predicted sales. The model can seperate the trend and weekly and yearly seasonality components well. In addition, it shows that the past sales (i.e. lagged sales) also have a strong predictive power for future sales. Last but not least, the promo-component impressively reveals that promotion can potentially increase sales by more than 1750 sales-units.

<table>
  <tr><td>
    <img 
        src="images/results_2.png"
        alt="Fashion MNIST sprite"  width="1000">
  </td></tr>
  <tr><td align="center">
    <b>Figure 2. Additive model components such as trend, saisonality, past sales and promo. 
  </td></tr>
</table>