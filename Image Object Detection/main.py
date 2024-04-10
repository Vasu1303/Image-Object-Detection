import cv2
import streamlit as st
import numpy as np

st.set_page_config(page_title = "Object Detector", page_icon="🌄")

classLabels=[]
classFile= 'coco.txt'
with open(classFile, 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')
    print(classLabels)


config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weights_path = 'frozen_inference_graph.pb'
img=cv2.imread('dogbike.jpeg')
model = cv2.dnn_DetectionModel(weights_path, config_file)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)  ## Gray Scale--> 255/2 = 127.5
model.setInputMean((127.5,127.5,127.5))   #Mean -- > [-1,1]
model.setInputSwapRB(True)











## MAKING THE WEBPAGE

st.title('Object Detection')
st.subheader("Mady by Vasu 😄")
uploaded_file = st.file_uploader("Upload the image that needs to be detected", accept_multiple_files=False, label_visibility='visible')
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    detected_image = cv2.imdecode(file_bytes, 1)
    st.text("Image Uploaded")
    st.image(opencv_image, channels="BGR", width=300)
    classIds ,  confs, bbox = model.detect(detected_image, confThreshold = 0.5)
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(detected_image, box, color = (0,0,255), thickness=2)
        cv2.putText(img,text=classLabels[classId-1].upper(), fontScale=0.8, color=(0,255,0) , thickness=1, org=(box[0]+10 , box[1]+35), fontFace= cv2.FONT_HERSHEY_COMPLEX)
        print(classLabels[classId-1].upper())
    
    

    # Now do something with the image! For example, let's display it:
    st.text("Image after Detection")
    st.image(detected_image, channels="BGR", width=300)
    st.write("-----------------------------------------------")

    st.write("Objects detected in the image are:")
    output=[]
    for i in classIds:
        if i not in output:
            
            st.write(classLabels[i-1].upper())

            output.append(i)
      





#cv2.imshow("Image", img)
#cv2.waitKey(0)




