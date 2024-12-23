# **LSTM Price Prediction**

## **Introduction**

This is my initial implementation of the LSTM (Long Short-Term Memory) model for price prediction. I am releasing this code for public use to help others utilize and build upon it. Additionally, I have used other models for similar projects, and their source code is available on my GitHub repository.

The model predicts the *Open*, *High*, *Low*, and *Close* prices for various financial assets, including indices, cryptocurrencies, commodities, and currency pairs. The following sections present the performance metrics of the model.

---

## **Results**

### **BTC-USD (Bitcoin)**

| Metric | Open | High | Low | Close |
| :---- | :---- | :---- | :---- | :---- |
| Mean Squared Error | 0.0008489510 | 0.0009795529 | 0.0010661283 | 0.0011113042 |
| Mean Absolute Error | 0.0239363016 | 0.0253854565 | 0.0252731072 | 0.0263076988 |
| R-squared | 0.9618445615 | 0.9568933098 | 0.9511450073 | 0.9510090004 |
| Median Absolute Error | 0.0209806155 | 0.0213325995 | 0.0199003023 | 0.0207255720 |
| Explained Variance Score | 0.9694517512 | 0.9652160792 | 0.9537688306 | 0.9563388589 |

### **![][image2]**

### **GC=F (Gold Futures)**

| Metric | Open | High | Low | Close |
| :---- | :---- | :---- | :---- | :---- |
| Mean Squared Error | 0.0102292391 | 0.0127224254 | 0.0086396166 | 0.0091730457 |
| Mean Absolute Error | 0.0858521072 | 0.0955619372 | 0.0768300133 | 0.0785435058 |
| R-squared | 0.5042878632 | 0.3761326958 | 0.5818894131 | 0.5498634680 |
| Median Absolute Error | 0.0786604464 | 0.0893523749 | 0.0630110043 | 0.0646407916 |
| Explained Variance Score | 0.8563785943 | 0.8134662843 | 0.8579252476 | 0.8404204121 |

### **![][image1]**

### **EURUSD (Euro/US Dollar)**

| Metric | Open | High | Low | Close |
| ----- | ----- | ----- | ----- | ----- |
| Mean Squared Error | 0.0004678367 | 0.0004778843 | 0.0007748126 | 0.0005051806 |
| Mean Absolute Error | 0.0179689022 | 0.0175163507 | 0.0231784207 | 0.0182077065 |
| R-squared | 0.8915972721 | 0.8914169532 | 0.8243772347 | 0.8831130975 |
| Median Absolute Error | 0.0148790617 | 0.0143152889 | 0.0205483317 | 0.0162848522 |
| Explained Variance Score | 0.9207051173 | 0.9007113662 | 0.8749229833 | 0.9102028001 |


### **![][image3]**

### **GSPC (S\&P 500 Index)**

| Metric | Open | High | Low | Close |
| :---- | :---- | :---- | :---- | :---- |
| Mean Squared Error | 0.0085264583 | 0.0081207582 | 0.0079891423 | 0.0076892149 |
| Mean Absolute Error | 0.0805443270 | 0.0776077142 | 0.0769452674 | 0.0745821802 |
| R-squared | 0.3710538626 | 0.4289697536 | 0.4101209639 | 0.4603227313 |
| Median Absolute Error | 0.0754465075 | 0.0712171335 | 0.0714640949 | 0.0656625427 |
| Explained Variance Score | 0.8424518479 | 0.8469419318 | 0.8395034957 | 0.8419850381 |


### **![][image4]**

---

## **Websites**

These results and similar projects have been a part of my experience working on predictive models. I have developed free tools that provide valuable insights into financial markets. You can access these tools at the following websites:

* [**Predict Price**](https://predict-price.com/)**:** Free AI-powered short-term (5/10/30 days) & long-term (6 months/1/2 years) forecasts for cryptocurrencies, stocks, ETFs, currencies, indices, and mutual funds.  
* [**Magical Prediction**](https://magicalprediction.com/)**:** Get free trading signals generated by advanced AI models. Enhance your trading strategy with accurate, real-time market predictions powered by AI.  
* [**Magical Analysis**](https://magicalanalysis.com/)**:** Discover free trading signals powered by expert technical analysis. Boost your forex, stock, and crypto trading strategy with real-time market insights.

---

## **Conclusion**

This LSTM implementation represents an initial effort to predict price movements across multiple asset classes. The model is open for public use, and I encourage everyone to build upon it. For further improvements and related projects, check out my GitHub repository.


[image1]: <https://raw.githubusercontent.com/taleblou/LSTM-Price-Prediction/refs/heads/main/Plot/LSTM_BTC-USD.png>
[image2]: <https://raw.githubusercontent.com/taleblou/LSTM-Price-Prediction/refs/heads/main/Plot/LSTM_GC%3DF.png>
[image3]: <https://raw.githubusercontent.com/taleblou/LSTM-Price-Prediction/refs/heads/main/Plot/LSTM_EURUSD%3DX.png>
[image4]: <https://raw.githubusercontent.com/taleblou/LSTM-Price-Prediction/refs/heads/main/Plot/LSTM_%5EGSPC.png>
