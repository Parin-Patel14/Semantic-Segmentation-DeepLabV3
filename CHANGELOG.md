#  Changelog  

##  Update 2 (Final Version) - Improved Model & Training  
###  **Key Improvements Over Update 1:**  
 **Dataset Handling**  
- Switched from **LabelMe annotations** to **RoboFlow with COCO JSON format**  
- Improved **preprocessing and data augmentation techniques**  

 **Model Architecture & Training**  
- Fine-tuned **DeepLabV3-ResNet50**, enabling auxiliary classifier layers  
- Introduced **hyperparameter tuning** for learning rate and weight decay  
- Implemented **checkpointing to save best model performance**  

 **Training Process & Optimization**  
- Replaced **fixed batch size and learning rate** with dynamic tuning  
- Added **SGD optimizer** as an alternative to Adam  
- Implemented **training vs. validation loss visualization**  

 **Model Evaluation & Visualization**  
- Added **Mean IoU, Pixel Accuracy, and Class Accuracy** metrics  
- **Overlayed segmentation masks** on images for better interpretability  
- Displayed **sample predictions vs. ground truth**  

 **Update 2 significantly improves dataset annotation, model training, and evaluation, leading to better accuracy and explainability.** 🚀  
