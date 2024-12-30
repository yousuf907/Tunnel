# Code for SHAP analysis and SHAP Slopes calculation

Python code to calculate SHAP Slopes presented in the paper What Variables Affect Out-Of-Distribution Generalization in Pretrained Models?

The code requires the SHAP package. You can install it from [here](https://shap.readthedocs.io/en/latest/).

To use the code, simply run the following:

```bash
python shap_slopes_calculation.py --data experiments_results.csv --metric 'Percentage OOD retained'
```

To get results for other metrics, change the metric argument. The choices are: 'Percentage OOD retained', 'Pearson Correlation', and 'ID OOD Alignment'
