import cv2
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.image import LoadImage
model_dir = '/Users/ycbd/photo/2013年/上高/小港/'
def clip_image(bboxes,i,filepath):
    '''
    filelist:文件夹路径
    i：批量保存的图片文件名，用数字表示
    im_path:图片路径
    '''

    result1 = os.path.split(filepath)
    file_name = os.path.splitext(result1[1])
    for i in range(len(bboxes)):
        bbox = bboxes[i].astype(np.int32)
        # file_path=os.path.join(im_path,'')
        im=cv2.imread(filepath)
        #[h,w]根据自己图片中目标的位置修改
        x1, y1, x2, y2 = bbox
        if y2-y1 > 80 and x2-x1 > 100:
            im = im[y1:y2,x1:x2]
            save_path = result1[0]+'/face/'
            save_path_file = os.path.join(save_path, file_name[0]+'_'+'face_'+str(i)+'.jpg')
            cv2.imwrite(save_path_file,im)
            i = i + 1

face_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet_facedetection_scrfd10gkps')
for root, dirs, files in os.walk(model_dir):
    for file in files:
        img_path = model_dir + file
        result = face_detection(img_path)
        # img = LoadImage.convert_to_ndarray(img_path)
        bboxes = np.array(result[OutputKeys.BOXES])
        if len(bboxes) > 0:
            clip_image(bboxes, 1,img_path)