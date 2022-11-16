# numpy >= 1.20
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_hrnetw48_human-wholebody-keypoint_image'
wholebody_2d_keypoints = pipeline(Tasks.human_wholebody_keypoint, model=model_id)
output = wholebody_2d_keypoints('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/keypoints_detect/img_test_wholebody.jpg')

# the output contains keypoints and boxes
print(output)