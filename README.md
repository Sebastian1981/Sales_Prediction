# Predict Bigmart´s Future Sales and Find Driving Features 

## Project Purpose
Predict sales for different Bigmart outlets using different machine learning regression models such as linear, tree and neural network models. Besides the sales forecast, we extract the feature driving sales by using a game-theoretical approach to make decision making of performant blackbox models transparent.
## Setup the Environment using Conda to run the JupyterNotebooks
- $conda create -n myenv python=3.8.11
- $conda active myenv
- $pip install -r requirements.txt

## Modeling Results
The figure below shows the historical sales from 2013 until July 2015 and the 14-day ahead forecast. Apparently, there is a good match beetween the forecast (blue curve) and the actual sales not only for the training period (blue dots) but also for the testing period (red dots).

<table>
  <tr><td>
    <img 
        src="images/results_1.png"
        alt="Fashion MNIST sprite"  width="1000">
  </td></tr>
    <tr><td align="center">
    <b>Figure 1. Historical sales data and future sales estimates derived by applying an additive regression model combined with deep learning. 
  </td></tr>
</table>

## Model Evaluation
The mean average prediction error is in the range of 4-6 % as depicted below.
<table>
  <tr><td>
    <img 
        src="images/results_3.png"
        alt="Fashion MNIST sprite"  width="1000">
  </td></tr>
  <tr><td align="center">
    <b>Figure 2. Cross-validated mean average error (mape). 
  </td></tr>
</table>

## Model Interpretability
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