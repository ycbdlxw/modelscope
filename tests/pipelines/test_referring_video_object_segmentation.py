# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ReferringVideoObjectSegmentationTest(unittest.TestCase,
                                           DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.referring_video_object_segmentation
        self.model_id = 'damo/cv_swin-t_referring_video-object-segmentation'

    @unittest.skip('skip since the model is set to private for now')
    def test_referring_video_object_segmentation(self):
        input_location = 'data/test/videos/referring_video_object_segmentation_test_video.mp4'
        text_queries = [
            'guy in black performing tricks on a bike',
            'a black bike used to perform tricks'
        ]
        start_pt, end_pt = 4, 14
        input_tuple = (input_location, text_queries, start_pt, end_pt)
        pp = pipeline(
            Tasks.referring_video_object_segmentation, model=self.model_id)
        result = pp(input_tuple)
        if result:
            print(result)
        else:
            raise ValueError('process error')

    @unittest.skip('skip since the model is set to private for now')
    def test_referring_video_object_segmentation_with_default_task(self):
        input_location = 'data/test/videos/referring_video_object_segmentation_test_video.mp4'
        text_queries = [
            'guy in black performing tricks on a bike',
            'a black bike used to perform tricks'
        ]
        start_pt, end_pt = 4, 14
        input_tuple = (input_location, text_queries, start_pt, end_pt)
        pp = pipeline(Tasks.referring_video_object_segmentation)
        result = pp(input_tuple)
        if result:
            print(result)
        else:
            raise ValueError('process error')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
