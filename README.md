# Project: Change Detection in Scenes with YOLOv8

## 📄 Description

This project aims to detect changes in scenes captured in two distinct datasets (`test.npz` and `base.npz`). Using image processing techniques and deep learning, we trained a YOLOv8 neural network to detect objects in the images and compare the detections between the scenes.

![Alt text](sample_app.png?raw=true "MyApp")

## 📂 Project Structure

```
/
├── train/
│   ├── pre_process/
│   │   ├── dataset/ # Images extracted from NPZ files
│   │   ├── images 
│   │   │   ├── labelled # Labeling files generated by labelImg
│   │   ├── export_imgs.py # Script to export images from NPZ
│   │   ├── splitTrainAndTest.py # Script divide dataset to train
│   │   ├── train.sh # ShellScript to train YOLOv8
│   ├── results_custom/  # YOLOv8 training results
│   │   ├── weights/
│   │   ├── confusion_matrix.png
│   │   ├── results.png
│   ├── results_default/  # Default YOLOv8 weights file
│   │   ├── weights/
├── server/
│   ├── static/ # Project static files: images, gifs, etc
│   ├── templates/  # Project HTML files
│   │   ├── index.html
│   ├── app.py  # Contains Flask information
│   ├── config.py  # Project configurations
│   ├── frame_processor.py  # Data processing
│   ├── object_detector.py  # Object detection with YOLOv8
│   ├── utils.py  # Utilities
├── play_yolo_simple.py  # Script for local display of detections
├── play_yolo_server.py  # Flask web service for displaying detections
├── play_yolo_streamlit.py  # Streamlit script for displaying detections
├── README.md
├── best.pt # File weights from YOLOv8
├── base.npz # Dataset 
├── test.npz # Dataset 
```

## 🚀 Environment Setup

### 1. Install Anaconda

If you don't have Anaconda installed yet, [download and install Anaconda](https://www.anaconda.com/products/distribution#download-section) to manage Python packages and environments.

### 2. Create a Virtual Environment

```bash
conda info --envs # show environments
conda env remove --name ktp # remove enviroment
conda create --name ktp pip python=3.9 # run on linux or windows
conda create --name ktp pip python=3.10.12 # run on mac os x
conda activate ktp
conda deactivate
```

### 3. Install Dependencies

Install the necessary libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure CUDA Environment

To train the YOLOv8 neural network, you need an environment with Ubuntu 20.04 and a GPU with CUDA installed. Follow the instructions from [NVIDIA to install CUDA](https://developer.nvidia.com/cuda-downloads).

It's ideal to have CUDA installed for detection as well, as it will make the detection process faster. If you don't have a GPU, it will be possible to run via CPU.

### 5. Download files

You need to download the files `base.npz`, `test.npz`, and `best.pt` for the project to work! These files are in a shared folder on OneDrive, which you can find in the Downloads section of this README.md. They are mandatory for the project to work! Download them and move them to the root of this project.

## 🏋️ Neural Network Training

Inside the `train` directory (see topic Downloads in this README.md), you should download the datasets (`base.npz` and `test.npz`) that contain the images for training and testing. The images were extracted from these files and saved in `.png` format. We used the `labelImg` tool to label the images before training the YOLOv8 neural network.

The training results, including model weights, confusion matrices, and other metrics, are available in the `results_custom` folder.

### Image Extraction and Labeling

1. Extract the images from the NPZ files and save them in `.png` format.
2. Use `labelImg` to label the images.

### Training with YOLOv8

To train the YOLOv8 model, follow the instructions in the [official YOLOv8 repository](https://github.com/ultralytics/yolov8).

## 🖥️ System Development

### Local Display with `play_yolo_simple.py`

The `play_yolo_simple.py` script displays object detections in windows of the operating system itself, allowing for quick and direct analysis of the results.

```bash
python play_yolo_simple.py
```

### Streamlit Display with `play_yolo_streamlit.py`

The `play_yolo_streamlit.py` script displays object detections in the browser, allowing for quick and direct analysis of the results via the web.

```bash
streamlit run play_yolo_streamlit.py
```

### Web Service with `play_yolo_server.py`

For a more accessible and interactive interface, use the web service implemented with Flask (`play_yolo_server.py`). This service allows the results to be accessed remotely through a web browser.

```bash
python play_yolo_server.py
```

Access the web interface at [http://localhost:5001](http://localhost:5001).

## 📝 Conclusion

This project demonstrates the application of deep neural networks for change detection in scenes captured by different datasets. Through the use of YOLOv8 and the Flask framework, we created a robust solution to identify and visualize changes in images efficiently and accessibly.

## ⚠️ Important Note

This README provides essential instructions for configuring and running the project. Make sure to follow all steps carefully to ensure that the environment is correctly configured and that all scripts work as expected. For more details on environment setup and tool usage, refer to the official documentation of the libraries and frameworks used.

## ⬇ Downloads

Download at the root of the project.

[base.npz](https://1drv.ms/u/s!ArAJOiCdbV7IirQHfobggiHMaDLPeA?e=Mvbgop).
[test.npz](https://1drv.ms/u/s!ArAJOiCdbV7IirQI1GNgD8O1Sek7oA?e=sZ5VFk).
[best.pt](https://1drv.ms/u/s!ArAJOiCdbV7IirQgYV9ZWmP0Xr_DRw?e=rRd133).

Download and Extract `train.zip` at the root of the project. Optional (only for train):

[train.zip](https://1drv.ms/u/s!ArAJOiCdbV7IirQ3_IVEV0kzmNYaVA?e=asyCQi).

---

We hope this project is helpful and provides a clear understanding of change detection in scenes using advanced deep learning techniques. For any questions or issues, feel free to open an issue or contact us.

🖥️👩‍💻🚀