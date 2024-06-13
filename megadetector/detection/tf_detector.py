"""

tf_detector.py

Module containing the class TFDetector, for loading and running a TensorFlow detection model.

"""

#%% Imports and constants

import numpy as np

from megadetector.detection.run_detector import \
    CONF_DIGITS, COORD_DIGITS, FAILURE_INFER
from megadetector.utils.ct_utils import truncate_float

import tensorflow.compat.v1 as tf

print('TensorFlow version:', tf.__version__)
print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())


#%% Classes

class TFDetector:
    """
    A detector model loaded at the time of initialization. It is intended to be used with
    TensorFlow-based versions of MegaDetector (v2, v3, or v4).  If someone can find v1, I 
    suppose you could use this class for v1 also.
    """
    
    #: TF versions of MD were trained with batch size of 1, and the resizing function is a 
    #: part of the inference graph, so this is fixed.
    #:
    #: :meta private:  
    BATCH_SIZE = 1


    def __init__(self, model_path):
        """
        Loads a model from [model_path] and starts a tf.Session with this graph. Obtains
        input and output tensor handles.
        """
        
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.Session(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')


    @staticmethod
    def __round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)


    @staticmethod
    def __convert_coords(tf_coords):
        """
        Converts coordinates from the model's output format [y1, x1, y2, x2] to the
        format used by our API and MegaDB: [x1, y1, width, height]. All coordinates
        (including model outputs) are normalized in the range [0, 1].

        Args:
            tf_coords: np.array of predicted bounding box coordinates from the TF detector,
                has format [y1, x1, y2, x2]

        Returns: list of Python float, predicted bounding box coordinates [x1, y1, width, height]
        """
        
        # change from [y1, x1, y2, x2] to [x1, y1, width, height]
        width = tf_coords[3] - tf_coords[1]
        height = tf_coords[2] - tf_coords[0]

        new = [tf_coords[1], tf_coords[0], width, height]  # must be a list instead of np.array

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.__round_and_make_float(d, precision=COORD_DIGITS)
        return new


    @staticmethod
    def __load_model(model_path):
        """
        Loads a detection model (i.e., create a graph) from a .pb file.

        Args:
            model_path: .pb file of the model.

        Returns: the loaded graph.
        """
        
        print('TFDetector: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Detection graph loaded.')

        return detection_graph


    def _generate_detections_one_image(self, image):
        """
        Runs the detector on a single image.
        """
        
        if isinstance(image,np.ndarray):
            np_im = image
        else:
            np_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(np_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # np_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)

        # performs inference
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: im_w_batch_dim})

        return box_tensor_out, score_tensor_out, class_tensor_out


    def generate_detections_one_image(self, 
                                      image, 
                                      image_id, 
                                      detection_threshold, 
                                      image_size=None,
                                      skip_image_resizing=False,
                                      augment=False):
        """
        Runs the detector on an image.

        Args:
            image (Image): the PIL Image object (or numpy array) on which we should run the detector, with
                EXIF rotation already handled.
            image_id (str): a path to identify the image; will be in the "file" field of the output object            
            detection_threshold (float): only detections above this threshold will be included in the return
                value
            image_size (tuple, optional): image size to use for inference, only mess with this
                if (a) you're using a model other than MegaDetector or (b) you know what you're
                doing
            skip_image_resizing (bool, optional): whether to skip internal image resizing (and rely on external 
                resizing).  Not currently supported, but included here for compatibility with PTDetector.
            augment (bool, optional): enable image augmentation.  Not currently  supported, but included 
                here for compatibility with PTDetector.

        Returns:
            dict: a dictionary with the following fields:
                - 'file' (filename, always present)
                - 'max_detection_conf' (removed from MegaDetector output files by default, but generated here)
                - 'detections' (a list of detection objects containing keys 'category', 'conf', and 'bbox')
                - 'failure' (a failure string, or None if everything went fine)
        """
        
        assert image_size is None, 'Image sizing not supported for TF detectors'
        assert not skip_image_resizing, 'Image sizing not supported for TF detectors'
        assert not augment, 'Image augmentation is not supported for TF detectors'
        
        if detection_threshold is None:
            detection_threshold = 0
            
        result = { 'file': image_id }
        
        try:
            
            b_box, b_score, b_class = self._generate_detections_one_image(image)

            # our batch size is 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections_cur_image = []  # will be empty for an image with no confident detections
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_entry = {
                        'category': str(int(c)),  # use string type for the numerical class label, not int
                        'conf': truncate_float(float(s),  # cast to float for json serialization
                                               precision=CONF_DIGITS),
                        'bbox': TFDetector.__convert_coords(b)
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = s

            result['max_detection_conf'] = truncate_float(float(max_detection_conf),
                                                          precision=CONF_DIGITS)
            result['detections'] = detections_cur_image

        except Exception as e:
            
            result['failure'] = FAILURE_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_id, str(e)))

        return result

    # ...def generate_detections_one_image(...)
    
# ...class TFDetector
