import cv2
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks
import matplotlib.pyplot as plt
import numpy as np

from modelscope.outputs import OutputKeys

def clip_image(detection_result,i,filename):
    '''
    filelist:文件夹路径
    i：批量保存的图片文件名，用数字表示
    im_path:图片路径
    '''
    bboxes = np.array(detection_result[OutputKeys.BOXES])
    for i in range(len(bboxes)):
        bbox = bboxes[i].astype(np.int32)
        # file_path=os.path.join(im_path,'')
        im=cv2.imread('srcImg.jpg')
        #[h,w]根据自己图片中目标的位置修改
        x1, y1, x2, y2 = bbox
        im=im[y1:y2,x1:x2]
        save_path = r'/Users/ycbd/PycharmProjects/modelscope/tests/images'
        save_path_file = os.path.join(save_path,filename+'_'+'face_'+str(i)+'.jpg')
        cv2.imwrite(save_path_file,im)
        i=i+1

face_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet_facedetection_scrfd10gkps')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_detection2.jpeg'
result = face_detection(img_path)
# if you want to show the result, you can run
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.preprocessors.image import LoadImage
img = LoadImage.convert_to_ndarray(img_path)
cv2.imwrite('srcImg.jpg', img)
img_draw = draw_face_detection_result('srcImg.jpg', result)
clip_image(result,1)
# show_image_object_detection_auto_result('/Users/ycbd/PycharmProjects/modelscope/tests/mytest/', result,'srcImg2.jpg')
plt.imshow(img_draw)
plt.axis('off')
plt.show()
