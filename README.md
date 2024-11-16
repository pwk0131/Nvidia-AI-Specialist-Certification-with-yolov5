# Nvidia AI Specialist Certification with yolov5


**If you'd like to give it a try, watch the Verbal Explanation Video inside the Guide.zip folder and follow along. Additional explanations can be found in the Readme file and the Notion Report.**

**Upload the yolov5_learning_example.ipynb file from the zip folder to Google Colab and use it there.**  

**If you'd like a more detailed and polished view, check out the Notion report -> 
<a href="https://island-lunge-8fe.notion.site/Nvidia-AI-Specialist-Certification-13b4e95f93fe80168053fc7688cde99b" target="blank">Go to Notion</a>**  


## **주제(Title)**

AI 객체 인식을 활용한 주방 내 위험 요소 탐지  
*Detection of Hazards in the Kitchen Using AI Object Recognition*  

## 개요(Introduction)  
이 시스템은 시각장애인, 아동, 그리고 주방 사용에 미숙한 사람들을 위해 개발하고자 한다. 주방에는 유리, 도자기, 날카로운 도구가 많아 시각적으로 안전을 판단하기 어려운 사용자나 미숙한 사용자들에게 특히 위험할 수 있다.

YOLOv5를 활용하여 이러한 위험 요소를 실시간으로 감지하고, 사용자에게 경고 신호를 제공함으로써 사고를 예방하고 안전한 환경을 조성한다. 또한 음성 안내를 통해 위험을 인식할 수 있도록 하여, 주방에서 보다 자율적이고 안전하게 작업하는 것을 목표로 한다.

*This system is intended for visually impaired individuals, children, and those inexperienced in using the kitchen. Kitchens contain numerous hazards, such as glass, ceramics, and sharp tools, which can be particularly dangerous for users who may find it difficult to visually assess safety or are inexperienced.*

*Using YOLOv5, this system aims to detect such hazards in real time, providing users with warning signals to prevent accidents and create a safer environment. Additionally, with audio guidance to help users recognize risks, the goal is to enable safer, more autonomous operation within the kitchen.*  

## 영상 취득 방법 (Video Acquisition Method)  

직접 주방을 촬영하여 취득했다. 총 다섯 개의 동영상을 촬영했으며  video1, video2, video3 는 AI학습에 사용하였고 물체를 다양한 각도로 자세히 관찰하는 것을 중점으로 촬영하였다. video4, video5 은 세 가지 동영상으로 학습된 AI의 성능을 검증하는 용도로 촬영하였다.

*I captured footage directly from the kitchen, recording a total of five videos. videos 1,2, and video 3 were used for AI training, focusing on observing objects from various angles in detail. video 4 and video 5 were recorded to test the performance of the AI trained on the first three videos.*  

Video for AI training  
&nbsp; [video 1](https://youtube.com/shorts/S8DIFcqSKOg?feature=share)  
&nbsp; [video 2](https://youtube.com/shorts/Ebps2Vl3PAA?feature=share)  
&nbsp; [video 3](https://youtube.com/shorts/UaPk0zsZzjg?feature=share)  

Video for validation  
&nbsp; [video 4](https://youtube.com/shorts/omjmIny2SNg?feature=share)  
&nbsp; [video 5](https://youtube.com/shorts/XNlsIMikUIA?feature=share) 

GoogleDrive  
&nbsp; [videos](https://drive.google.com/drive/folders/1GNmb12JI663gPk5rnA_XMYyvPFn5h24l?usp=drive_link)

## 프로젝트 진행(Project Progress)

### 1. VABMIX_2  (Video editing tool)

영상은 640x640 해상도로 제작하였으며, 해상도 조절 도구로는 '뱁믹스2'를 사용했다.

**The video was created with a resolution of 640x640, and 'VABmix2' was used as the resolution adjustment tool.*

![화면_캡처_2024-11-11_213738](https://github.com/user-attachments/assets/e1127b75-6a48-4311-baba-f4b44ccd45b7)  

---

### 2. 파일 설정하기 (Setting File)  
편집한 영상의 이미지 라벨링은 [DarkLabel](https://github.com/darkpgmr/DarkLabel) 을 사용하였다.

라벨링을 하기 위해서는 `darklabel.yml` 파일을 수정해야 한다.

주방 내 위험 요소로는  깨질 가능성이 있는 “그릇”, “날카로운 ‘가위”, “칼” 을 라벨링 하기 위해 `risk_classes`를 생성 후 내부 요소는 `["bowl","scissors","knife"]`로 설정했다.

이후 `format_10` 에 설정 후 진행 할 것 이다.

*The image labeling for the edited videos was done using [DarkLabel](https://github.com/darkpgmr/DarkLabel).*

*To perform the labeling, the `darklabel.yml` file needs to be modified.*

*For labeling the kitchen hazards, such as breakable "bowl", sharp "scissors", and "knife", a `risk_classes` section was created with the elements set to `["bowl", "scissors", "knife"]`.*

*After that, the labeling will be carried out by setting it in `format_10`.*  

![스크린샷_2024-11-13_101356](https://github.com/user-attachments/assets/da2bfa78-744a-4eee-a8e4-ae4d9e3c958c)  

![스크린샷_2024-11-13_112902](https://github.com/user-attachments/assets/ec320967-eda5-4514-95b9-e637e59d9a4b)

  

`darklabel.yml` 파일을 설정했다면, 이후 모델 학습에 필요한 `data.yaml` 파일도 설정해야 한다.

`data.yaml` 파일은 `darklabel.yml` 파일에서 설정한 요소를 나열하고, 모델 학습에 필요한 파일의 경로를 지정한다. 마지막으로 요소의 개수를 기입하여 설정을 완료한다.

*Once you have configured the `darklabel.yml` file, you will also need to set up the `data.yaml` file, which is essential for model training.*

*The `data.yaml` file should list the elements defined in `darklabel.yml` and specify the file paths necessary for model training. Finally, complete the configuration by indicating the total number of elements.*  

![스크린샷_2024-11-12_092458](https://github.com/user-attachments/assets/06dbac8c-ce1a-42a0-9633-6ed8d108e3af)  

---

### 3. __DarkLabel 사용하기 (using DarkLabel)__
DarkLabel을 실행하여 직접 취득한 영상을 “Open video” 버튼을 클릭하여 화면에 띄운다.

앞서 언급한 세 가지 비디오들을 선택하였다.

“As images” 버튼을 클릭하여 비디오를 이미지로 변환한 후, `Train/images` 폴더에 저장하였다. 이 과정을 통해 총 982개의 이미지가 생성되었다.

미리 설정한 `format10(risk_classes)`를 선택하면 주방 내 위험요소인 `["bowl", "scissors", "knife"]`가 정상적으로 표시되는 것을 확인할 수 있다.

이제 각 이미지를 순차적으로 라벨링 하면 된다. 

*Run DarkLabel and click the "Open Video" button to display the video on the screen.*

*Select the video from this link: *

*Next, click the "As Images" button to convert the video into images and save them in the `Train/images` folder. This process generates a total of 982 images.*

*Select the pre-configured `format10(risk_classes)` to confirm that the kitchen hazards—`["bowl", "scissors", "knife"]`—are correctly displayed.*

*Now begin labeling each image sequentially.*

[images](https://drive.google.com/drive/folders/1c5ISbHS8qT-4rgjubeotzT8SvsICsifp?usp=drive_link) - go to GoogleDrive  

 ![화면 캡처 2024-11-13 190335](https://github.com/user-attachments/assets/80d417df-19b5-4d19-9ba0-2651b79997a4)

---

### 4. 라벨링 결과 (Result of Labeling )  

[classid, ncx, ncy, nw, nh]  

// classid: Object class ID (integer)  
// ncx, ncy: Normalized x and y coordinates of the bounding box center (between 0 and 1)  
// nw, nh: Normalized width and height of the bounding box (also between 0 and 1)  

각 라벨 파일은 `[classid, ncx, ncy, nw, nh]` 형식의 "txt" 파일로 저장되며, 여기서 `classid`는 객체의 클래스 ID, `ncx`와 `ncy`는 바운딩 박스 중심의 x, y 좌표를 나타내고, `nw`와 `nh`는 바운딩 박스의 너비와 높이를 의미한다.

이 데이터는 `yolov5` 모델을 학습하는 데 사용될 예정이다.

*Each label file is saved in the format `[classid, ncx, ncy, nw, nh]` as a "txt" file, where `classid` represents the object class ID, `ncx` and `ncy` represent the x and y coordinates of the bounding box center, and `nw` and `nh` represent the width and height of the bounding box.*

*This data will be used for training a `yolov5` model.*

[labels](https://drive.google.com/drive/folders/1cEuR1P9Xbd9wEla88cRfPvMtj98iXEhM?usp=drive_link) - go to GoogleDrive  

![화면 캡처 2024-11-13 190614](https://github.com/user-attachments/assets/b465e2f7-9d03-4824-b3f9-d942f4f4547c)  

---

### 5. yolov5 모델 학습 시작(Start training yolov5 model) feat colab  

#### 5-1. 구글 코랩 연결(Connecting Colab)

[colab](https://colab.google/)을 실행한 후 자신의 구글 드라이브와 연결한다.

*After running [colab](https://colab.google/), connect it to your Google Drive.*  

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive #Go to the path your set, this is example
%pwd
```

#### 5-2. yolov5 repository clone  
[yolov5](https://github.com/ultralytics/yolov5) 레포지토리를 클론 후 설치한다.

*Clone the [yolov5](https://github.com/ultralytics/yolov5) repository and install it.*  

```python
# start clone yolov5
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
%pip install -qr requirements.txt
```

![화면_캡처_2024-11-12_012808](https://github.com/user-attachments/assets/496fc324-c2cf-4ee6-aae3-79adfd38a5d2)  

---

#### 5-3. _preproc  and Create_npy  

```python
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.python.eager.context import eager_mode

def _preproc(image, output_height=512, output_width=512, resize_side=512):
    ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h*scale), int(w*scale)])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
        return tf.squeeze(cropped_image)

def Create_npy(imagespath, imgsize, ext) :
    images_list = [img_name for img_name in os.listdir(imagespath) if
                os.path.splitext(img_name)[1].lower() == '.'+ext.lower()]
    calib_dataset = np.zeros((len(images_list), imgsize, imgsize, 3), dtype=np.float32)

    for idx, img_name in enumerate(sorted(images_list)):
        img_path = os.path.join(imagespath, img_name)
        try:
            if os.path.getsize(img_path) == 0:
                print(f"Error: {img_path} is empty.")
                continue

            img = Image.open(img_path)
            img = img.convert("RGB")
            img_np = np.array(img)

            img_preproc = _preproc(img_np, imgsize, imgsize, imgsize)
            calib_dataset[idx,:,:,:] = img_preproc.numpy().astype(np.uint8)
            print(f"Processed image {img_path}")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    np.save('calib_set.npy', calib_dataset)

```

`_preproc` 함수는 이미지를 크기 조정하고 중앙 부분을 잘라내는 작업을 수행한다.

입력된 이미지의 세로(`h`), 가로(`w`) 크기를 기준으로 작은 쪽이 `resize_side`와 일치하도록 비율을 유지하면서 크기를 조정한다.

`output_height`와 `output_width`로 설정된 크기로 중앙을 잘라낸다.

이 함수는 처리된 이미지를 `TensorFlow` 텐서 형식으로 반환한다.

`Create_npy` 함수는 주어진 디렉토리(`imagespath`)에서 특정 확장자(`ext`)를 가진 모든 이미지를 불러온다.

각 이미지 파일을 열고 전처리 후 `calib_dataset`라는 배열에 저장한다.

- 이미지 파일의 크기가 `0`인 경우 오류 메시지를 출력하고 해당 이미지를 건너뛴다.
- 이미지가 `RGBA` 또는 다른 형식일 경우, `RGB`로 강제 변환해 처리한다.
- 전처리된 이미지를 `uint8` 형식으로 변환하여 배열에 저장한다.

최종적으로, `calib_set.npy` 파일에 모든 이미지 데이터를 저장한다.

*The `_preproc` function performs resizing and central cropping on an image.*

*Based on the image’s height (`h`) and width (`w`), it scales the image so that the smaller dimension matches the `resize_side` length while maintaining the aspect ratio.*

*Then, it crops the image to the size specified by `output_height` and `output_width`.*

*This function returns the processed image in TensorFlow tensor format.*

*The `Create_npy` function loads all images with a specific extension (`ext`) from a given directory (`imagespath`).*

*It opens each image file, applies preprocessing, and stores the processed image in an array called `calib_dataset`.*  

- *If the image file size is `0`, it outputs an error message and skips the file.*  
- *If the image format is `RGBA` or another format, it forcibly converts it to `RGB` for processing.*  
- *The preprocessed image is converted to `uint8` format before being stored in the array.*  

*Finally, it saves all image data to a `calib_set.npy` file.*  

---

#### 5-4. 모델 학습하기 (Train the model)  
```python
# Training
!python train.py  --img 640 --batch 16 --epochs 300 --data /content/drive/MyDrive/yolov5/data.yaml --weights yolov5n.pt --cache
```  

이 코드는 YOLOv5 모델을 훈련하기 위한 명령어 이다.

각 매개변수는 모델의 훈련 설정을 정의하며, 아래와 같은 의미를 갖는다.

- `!python train.py`: YOLOv5의 `train.py` 파일을 실행하여 모델 훈련을 시작한다.
- `-img 640`: 입력 이미지의 크기를 640x640 픽셀로 설정한다.
- `-batch 16`: 배치 크기를 16으로 설정하여 한 번에 16개의 이미지를 훈련에 사용한다.
- `-epochs 300`: 총 300번의 epochs 동안 모델을 훈련 시킨다.
- `-data /content/drive/MyDrive/yolov5/data.yaml`: 데이터 세트에 대한 설정 파일 경로이다. 이 파일에는 클래스 정보, 훈련 및 검증 이미지 경로 등이 포함된다.

즉, `data.yaml` 파일의 내용을 바탕으로 모델 학습이 진행된다. 이 과정은 약 2시간 정도 소요되었다.

*This command is used to train the YOLOv5 model. Each parameter defines a setting for the model training, with the following meanings:*

- *`!python train.py`: Runs the YOLOv5 `train.py` file to start model training.*
- *`-img 640`: Sets the input image size to 640x640 pixels.*
- *`-batch 16`: Sets the batch size to 16, meaning 16 images are used per training step.*
- *`-epochs 300`: Trains the model over a total of 300 epochs.*
- *`-data /content/drive/MyDrive/yolov5/data.yaml`: Specifies the path to the dataset configuration file, which includes class information and the paths for training and validation images.*

*In other words, the model training proceeds based on the contents of the `data.yaml` file. This process took approximately 2 hours.*  


![화면_캡처_2024-11-12_015103](https://github.com/user-attachments/assets/27c52b70-c484-4c08-942d-4bbe644f50c2)  

---

## 결과(Results)  

### 1. 텐서보드(TensorBoard)  

```python
# view Results from TensorBoard
%load_ext tensorboard
%tensorboard --logdir runs
```

![화면_캡처_2024-11-12_071447](https://github.com/user-attachments/assets/e42da432-973a-4ec2-adca-08ec6b7594a6)  

Jupyter Notebook에서 TensorBoard를 실행하여 결과를 확인한다. TensorBoard dashboard가 열리면 모델 훈련의 진행 상황, 손실 그래프, 정확도 등의 지표를 시각적으로 확인할 수 있다.

*By running TensorBoard in Jupyter Notebook, you can check the results. Once the TensorBoard dashboard opens, you can visually monitor metrics such as training progress, loss graphs, and accuracy.*  

### 2. F1_curve

![F1_curve](https://github.com/user-attachments/assets/ba78bfe6-abad-4e61-b687-ab96859fcd59)  

### 3. P_curve /  R_curve  

![P_curve](https://github.com/user-attachments/assets/14115d7f-70cb-4e52-a4d0-81e80c988025)
![R_curve](https://github.com/user-attachments/assets/791203b0-0988-4c26-abaf-31f9c25f8e10)  

### 4. confusion_matrix  

![confusion_matrix](https://github.com/user-attachments/assets/9342247c-e9c4-4f1e-b517-6563428e82d7)  

### 5. Result  

![results](https://github.com/user-attachments/assets/9be1634a-6eff-4a84-af0b-4c811070f783)  

### 6. label batch

![val_batch2_labels](https://github.com/user-attachments/assets/6189b6cd-d730-4098-bcf4-7620162d1a7b)

링크를 통해 더 많은 결과를 확인 가능 

*More results can be viewed through the file on the right.* [EXP2](https://drive.google.com/drive/folders/12_YXtZsEaEA4LU8m3a0eXbQDVaupZp4D?usp=sharing) - go to GoogleDrive  

---

## 탐지(Detect)  
`yolov5`의 학습이 완료되었다면,  학습에 사용한 이미지와 비디오들을 감지(detect)한다.

*Once the training of `yolov5` is complete, detect objects in the images and videos used for training.*  

- video detect method
  ```python
  #This path varies depending on the user.
  !python detect.py --weights runs/train/exp2/weights/best.pt --source /content/drive/MyDrive/yolov5/video/video_1.mp4
  ```
  
- video detect output
  ![스크린샷_2024-11-12_095613](https://github.com/user-attachments/assets/8cc09922-17ca-428a-9423-5d1b919b65d9)

Detect_Result video  
&nbsp; [Verified video 1](https://youtube.com/shorts/Uke-PBQynwc?feature=share)  
&nbsp; [Verified video 2](https://youtube.com/shorts/c2rb6TWAViE?feature=share)  
&nbsp; [Verified video 3](https://youtube.com/shorts/jicgc_UsZGw?feature=share)  
&nbsp; [Verified video 4](https://youtube.com/shorts/V1xRYH_J4rQ?feature=share)  
&nbsp; [Verified video 5](https://youtube.com/shorts/YXJ2HtT1iWM?feature=share)  

[detect](https://drive.google.com/drive/folders/13wFPaY0pXwfUoBeJnAvXxmPbxGpWvlG0?usp=drive_link) - go to googleDrive  

---

Image Detect Result

학습에 사용되었던 이미지 982개 전체에 대해 객체 탐지를 수행하였다.

그 중 임의로 선택한 2개 이미지의 탐지 결과를 아래에 나열하였다.

*Object detection was performed on all 982 images used for training.*

*The detection results of six randomly selected images from this set are listed below.* 
  
![8b5cd22e-0770-43b8-b9a6-b84049b3e429](https://github.com/user-attachments/assets/34e8fc35-2246-4b5e-9b4d-73540ca81fc9)
![f90137f4-5f3d-49c8-8bf2-54fd9af370da](https://github.com/user-attachments/assets/c5562c11-b089-43af-b548-848977073463)  

[see more](https://drive.google.com/drive/folders/13xB80xuIFD0sUOan4zkytTEJFSH8nIzC?usp=drive_link) - go to googleDrive  
[see more notion](https://island-lunge-8fe.notion.site/Nvidia-AI-Specialist-Certification-13b4e95f93fe80168053fc7688cde99b) - go to notionReport


---

# 마무리(Wrap up)

모든 위험 요소 객체의 `Confidence Score` 는  높은 신뢰도 값을 가지고 있어, 같은 방식으로 주방 내 다른 위험 요소들도 높은 신뢰도 값으로 검출할 수 있음을 의미한다. 예를 들어 뜨거운 물체로 판단될 수 있는 냄비, 프라이팬 등과 같은 위험 요소들을 추가하여 모델을 더욱 발전시킬 수 있다.

또는 다양한 각도에서 모양이 크게 다른 그릇,칼,가위의 학습량을 늘려 현재보다 더더욱 높은 신뢰도 값을 지 향 하는 것 또한 방법이다.

시각장애인, 아동 등.. 주방 사용에 미숙한 이들이 최소한 심각한 상처 만은 받지 않도록 더 나아가 안전한 주방이 되도록 기대하는 마음으로 주제를 마무리한다.

*The `confidence scores` for all hazardous objects are high, indicating that other potential hazards in the kitchen can also be detected with high confidence using the same approach. For instance, additional hazardous items like pots and frying pans, which may be identified as hot objects, can be added to further enhance the model.*

*Alternatively, increasing the training volume for items like bowls, knives, and scissors, which vary greatly in shape from different angles, could be another approach to aim for even higher reliability than at present.*

*In conclusion, with the hope of creating a safer kitchen environment so that visually impaired individuals, children, and others unfamiliar with kitchen use can avoid serious injuries, we wrap up this topic.*  

---

Thank you for reading until the end
