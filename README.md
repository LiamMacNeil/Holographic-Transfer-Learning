# Holographic-Transfer-Learning
Classifying holographic plankton images with 4 pre-trained, state-of-the-art convolutional neural networks. Includes stratified k-fold cross validation, evaluation metrics, and plotting tools.

Training and test image folders are seperate pandas dataframes. Each model will spin up the desired number of training epochs and stratified folds for training:validation, followed by plot training:validation results for log loss and threshold metrics including standard deviation by fold. Epochs and folds can be changed as global variables. Evaluation on the test set with be in batches, averaged, and with standard deviations. Precision-recall curves are then generated for each class, comparing the trade-off between precision and recall at every decision threshold, and the area under each curve (calculated as average precision- see Davis and Goadrich, 2006; Ferri et al. 2009), summarize the overall classification performance for each class.  

The momentum values in Batch normalization layers in ResNet50V2, InceptionV3, and Xception are reduced from Tensorflow defaults 0.99, to 0.9. Apologies for the blunt and repetitive objects for plotting.

Davis J, Goadrich M. The relationship between Precision-Recall and ROC curves. Proceedings of the 23rd International Conference on Machine Learning. 2006; 233–240. 

Ferri C, Hernández-Orallo J, Modroiu R. An experimental comparison of performance measures for classification. Pattern Recognit Lett. 2009; 30(1): 27–38. 
