# II_Group1_Lab2
Code for Image Interpretation Lab 1 at ETH Zurich FS21.  
Team member: **Liyuan Zhu, Rushan Wang, Nianfang Shi**  

The goal of Lab 2 is to design hand crafted features of satellite images and utilize the hand crafted features to regress the canopy height.  

We use R,G,B,NIR,NDVI,MSAVI, VARI, ARVI, GCI, SIPI, HSV, LAB features as our input.  

The feature extraction and combination are implemented in Hand_crafted_features.py  

We use XGBoost for regression.  
___

Results with different learning rate.
| lr       | 0.5|0.1|0.05|0.03|0.01
| -------- | ----- | ----| -----|-----|-----|
|RMSE      |8.617768 | 8.219682|8.01|7.9315944|8.23

Results with different max depths
| max_depth  |2|3|4|5|6
| -------- | ----- | ----| -----|-----|-----|
|RMSE      |8.04| 7.999|8.01|8.01|8.02




