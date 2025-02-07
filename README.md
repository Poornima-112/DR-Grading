# Diabetic Retinopathy Grading with Multi-Task Learning

## Description
Developed an end-to-end machine learning pipeline for the classification and grading of Diabetic Retinopathy (DR) using the [IDRiD dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid), leveraging deep learning techniques to predict DR severity.  
**Diabetic Retinopathy (DR)** is a diabetes-related eye disease that can lead to blindness if left untreated. Early detection is crucial to prevent vision loss, as timely intervention can significantly reduce the risk of severe complications. DR is classified into five stages:  
1. **No DR**  
2. **Mild DR**  
3. **Moderate DR**  
4. **Severe DR**  
5. **Proliferative DR**

### Key Components:
- **Classification Model:**  
  Employed a ConvNeXt model for DR grading, trained on retinal fundus images to classify DR severity into the five stages outlined above.  
  [Why ConvNeXt ?](https://arxiv.org/abs/2201.03545)

- **Streamlit Interface:**  
  Built an interactive web application using Streamlit to visualize model predictions, displaying DR grades and providing real-time predictions on new retinal images.  
  [Link to Streamlit Web Application](https://dr-grading-dl.streamlit.app/)
  
### Detection Results:
Here are the detection results for each of the five classes, showcasing the model's performance across various stages of DR:
- **No DR:**  
  ![Screenshot 2025-01-10 115618](https://github.com/user-attachments/assets/a4d435a1-a7f9-4fd4-9ef1-c16bfe486cd8)
 
- **Mild DR:**
 ![Screenshot 2025-01-14 202458](https://github.com/user-attachments/assets/78070253-eb80-45f6-8b99-69661d1a2707)
 
- **Moderate DR:**  
![Screenshot 2025-01-14 202019](https://github.com/user-attachments/assets/7392c4ea-6b0c-4fef-96b5-405cae9f55ff)

- **Severe DR:**  
 ![Screenshot 2025-01-14 211602](https://github.com/user-attachments/assets/8ba948b6-3850-4cff-954c-810e33d6b19f)

- **Proliferative DR:**  
![Screenshot 2025-01-14 202103](https://github.com/user-attachments/assets/154ca34d-3a78-404c-b374-5912827f7a8f)


### Future Work:
- **Segmentation:**  
  Integrate a UNet++ architecture for the segmentation of DR-related lesions and optic disc localization.
  [Why UNet++ ?](https://arxiv.org/abs/1807.10165)
  
- **Localization:**  
  Implement localization tasks for identifying optic disc and fovea center locations using the IDRiD dataset.
  
- **Interpretability:**  
  Investigate interpretability techniques like Grad-CAM and attention maps to provide insights into model decisions by highlighting lesion areas contributing to DR severity grades.

