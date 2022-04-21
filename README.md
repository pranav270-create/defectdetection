# defectdetection
Automated Defect Detection and Reinforcement Learning based robot arm removal

Kaggle Data Set Link: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
Note: Use unaugmented 512x512 images

- Data Set Cleaning is a notebook that provides methods for visualization and normalization
- DCGAN is a notebook with Generator and Discriminator that acheives min FID = 64 when trained for 500 epochs with augment = True and p = 0.8.
- PreTrained Classifier has an Augmentation Class for data set improvement and uses Transfer Learning from VGG16 to acheive high validation accuracy up to D = 100.
- Final BNN does unsupervised learning on the data set to break into a 3 class problem as well as Sequential 2 class problems. When compared to the loaded model, it outperforms on sequential learning after 20 epochs of switching tasks.
- Hough Transform is a notebook that creates a custom data set (D = 1000) for parameter fine tuning, p1 p2 blur minR and maxR as well as calculates the relevant metrics of num misses, inference time, and average IOU score.
- Full Sequence reads in a blank or populated image from the RealSense Viewer and tests the whole pipeline by outputting an action for the Franka Arm to move to.

- Networks.py, Hough.py, CNNClassifier.py are all depreciated files that were used in earlier iterations.
