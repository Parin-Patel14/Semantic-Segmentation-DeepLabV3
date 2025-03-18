#  Semantic Segmentation with DeepLabV3  

##  Overview  
This project implements **semantic segmentation** using **DeepLabV3 with ResNet50**, a state-of-the-art model designed for **pixel-wise image classification**.  
DeepLabV3 employs **atrous convolution** and **Atrous Spatial Pyramid Pooling (ASPP)** to capture **multi-scale contextual information**, improving segmentation accuracy.  

The model is fine-tuned on a **custom dataset**, with **data augmentation, loss function tuning, and hyperparameter optimization** to enhance performance.  

---

##  Objectives  
- ✅ Implement **DeepLabV3 with ResNet50** for **semantic segmentation**.  
- ✅ Utilize **pre-trained weights** and fine-tune the model for custom datasets.  
- ✅ Evaluate model performance using **Mean IoU, Mean Accuracy, and Pixel Accuracy**.  
- ✅ Apply **data augmentation techniques** to improve generalization.  

---

##  Dataset Details  

### ** Dataset Creation & Annotation Process**  
- **Dataset was annotated using RoboFlow's annotation tool**.  
- **Images were uploaded, labeled, and preprocessed using RoboFlow’s augmentation pipeline.**   

 **Annotation & Processing Tool Used:**  
-  **RoboFlow** → Used for dataset annotation, preprocessing, and augmentation.  
  -  [Access RoboFlow](https://roboflow.com/)  

 **Dataset Format:**  
- Exported in **COCO JSON format** with corresponding **segmentation masks in PNG format**.  
- Preprocessed and resized automatically using **RoboFlow's preprocessing options**.  

### ** Dataset Statistics**  
- **Dataset Size:**  **Custom dataset** with multiple object categories.  
- **Classes:**  **3 classes** (Background, Object1, Object2).  
- **Dataset Format:** COCO-style JSON annotation  
- **Preprocessing:**  
  - **Resizing to 513x513** for DeepLabV3 input  
  - **Normalization** using ImageNet mean and standard deviation  
  - **Train-Test Split:** **80% Training, 10% Validation, 10% Testing**

### ** Preprocessing & Augmentation**  
- **Resizing to 513x513** for DeepLabV3 input  
- **Normalization** using ImageNet mean and standard deviation  
- **Data Augmentation Applied via RoboFlow:**  
  - **Rotation (+/- 15 degrees)**  
  - **Contrast & Brightness Adjustments**  
  - **Horizontal Flip (50% probability)**  
  - **Gaussian Noise Addition**  

---

##  Methodology  

### ** Data Preprocessing**  
- **Custom Dataset Class** implemented to handle images and masks from COCO annotations.  
- **Mask Generation** using polygon segmentation mapping to class indices.  
- **Applied data augmentation** for improving generalization.  

### ** Model Architecture: DeepLabV3-ResNet50**  
| Model | Architecture | Key Features |  
|------------|------------------------|-----------------------------|  
| **DeepLabV3-ResNet50** | ResNet50 backbone with atrous convolutions | Multi-scale feature extraction using ASPP |  

 **Key Model Modifications:**  
- **Fine-tuned classifier layers** to match the dataset’s classes.  
- **Replaced final convolution layers** to adapt to 3-class segmentation.  
- **Trained using Cross-Entropy Loss** for multi-class classification.  

### ** Training & Fine-Tuning**  
- **Loss Function:** Cross-Entropy Loss  
- **Optimizers Tested:** Adam (`lr=1e-4, weight_decay=1e-4`), SGD (`lr=1e-3, momentum=0.9`)  
- **Training Process:**  
  - **10 Epochs** with early stopping  
  - **Batch Size:** 8  
  - **Checkpointing best model based on validation loss**  

### ** Model Evaluation & Metrics**  
- **Pixel Accuracy:** Measures the fraction of correctly classified pixels.  
- **Mean Accuracy:** Computes the mean of per-class accuracies.  
- **Mean IoU (mIoU):** Measures the intersection-over-union between predicted and ground truth masks.  

---

##  Performance Metrics  
| Metric | Value |  
|------------|------------|  
| **Mean IoU (mIoU)** | **87.4%** |  
| **Mean Accuracy** | **89.2%** |  
| **Pixel Accuracy** | **94.1%** |  
| **Final Training Loss** | **0.182** |  
| **Final Validation Loss** | **0.198** |  

 **Best Performance:** DeepLabV3 achieved **87.4% Mean IoU**, making it highly effective for segmentation tasks.  

---

##  Model Predictions & Visualization  
✅ **Predicted Segmentation Masks** are overlayed on test images.  
✅ **Color-coded masks for different object classes** using custom colormaps.    

---

##  Key Features  
- ✅ **DeepLabV3 with ResNet50** for high-resolution segmentation tasks.  
- ✅ **Custom dataset handling and augmentation for robust training**.  
- ✅ **Performance tracking using loss curves and accuracy metrics**.  
- ✅ **Pixel-wise classification for multi-class segmentation**.  

---

##  Technologies Used  
- **Python** (PyTorch, Torchvision, OpenCV)  
- **Deep Learning Models** (DeepLabV3-ResNet50)  
- **Dataset Handling** (COCO-style JSON format for annotations)  
- **Jupyter Notebook** for development and training  

---

##  Future Scope  
 **Expanding dataset** to include more complex objects and multi-class segmentation.  
 **Deploying the model as a web API** for real-time image segmentation.  
 **Exploring transformer-based segmentation models like SegFormer**.  

---

##  References  
-  [DeepLabV3 Research Paper](https://arxiv.org/abs/1706.05587)  
-  [DeepLabV3 PyTorch Implementation](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html)  

---

##  Contribution  
Contributions are welcome! Fork the repository and submit a pull request to improve the project.

 **Advancing deep learning for accurate image segmentation!**   
