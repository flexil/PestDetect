# AI-Powered Pest Detection Application


# Overview


This application utilizes YOLOv11, a state-of-the-art object detection algorithm, to detect pests and diseases on tomato leaves. The model was trained on a large dataset of tomato leaf images and achieves impressive results.


# Model Performance


The YOLOv11 model was evaluated on a test dataset after 50 epochs with a batch size of 100. The evaluation metrics are:


- Precision: 0.3671
- Recall: 0.2634
- mAP@50: 0.2814
- mAP@50-95: 0.1429
- Fitness: 0.1567


These metrics demonstrate the model's ability to accurately detect pests and diseases on tomato leaves.


# Features


- Real-time detection: Detect pests and diseases in real-time using the YOLOv11 algorithm.
- High accuracy: Achieves high precision and recall rates.
- Tomato leaf dataset: Trained on a large dataset of tomato leaf images.
- User-friendly interface: Easy-to-use interface for uploading images and viewing detection results.


# Requirements


- Python 3.12
- Streamlit
- Ultralytics YOLOv11
- OpenCV


# Usage


1. Clone the repository.
2. Install required dependencies in the requirements.txt file
3. Run the application using streamlit using streamlit run app.py  commad on your linux ternimal.
4. Upload an image of a tomato leaf.
5. View detection results.


Future Development


- Improve model performance by increasing dataset size and training epochs.
- Integrate with other computer vision algorithms for enhanced detection.
- Develop a mobile application for farmers and agricultural professionals.
