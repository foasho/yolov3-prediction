# -*- coding: utf-8 -*-
import colorsys
import numpy as np
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import keras.backend as K
import sys

basepath = os.getcwd().replace("\\", "/")
model_path = basepath + '/prediction/target_model.h5'
anchors_path = basepath + '/prediction/yolo_anchors.txt'
classes_path = basepath + '/prediction/voc_classes.txt'
model_image_size = (224, 224)
font_path = basepath + "/font/FiraMono-Medium.otf"

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def resizePilImage(pilimage, width):
    resize_image = pilimage.resize((width, int(width * pilimage.size[1] / pilimage.size[0])))
    return resize_image


class YOLO(object):
    _defaults = {
        "model_path": model_path,
        "anchors_path": anchors_path,
        "classes_path": classes_path,
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": model_image_size,
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        #複数のGPUを利用する場合
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        #BOX(座標)、SCORE(打倒率)、CLASS(ラベル名)を取得する
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font=font_path,
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        #描画する画像データを複製しておく
        drawimage = image.copy()

        imagels = []
        classnamels = []
        coordinate_ls = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(drawimage)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))
            xmin, ymin, xmax, ymax = left, top, right, bottom
            coordinate = {"label": predicted_class,
                          "score": score,
                          "xmin": xmin,
                          "ymin": ymin,
                          "xmax": xmax,
                          "ymax": ymax}
            coordinate_ls.append(coordinate)

            #画像切り取り
            cvimg = pil2cv(image)
            cvimgcut = cvimg[top:bottom, left:right]
            cvpilimgcut = cv2pil(cvimgcut)
            imagels.append(cvpilimgcut)
            classnamels.append(predicted_class)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, predicted_class, fill=(0, 0, 0), font=font)
            del draw

        return imagels, classnamels, drawimage, coordinate_ls

    def close_session(self):
        self.sess.close()

def imshow(cv2image):
    cv2.imshow("test", cv2image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectrun(pil_image):
    pil_image = resizePilImage(pil_image, 1200)
    yolo_data = YOLO()
    detectimagels, classnamels, drawimage, coordinate_ls = yolo_data.detect_image(pil_image)
    return detectimagels, classnamels, drawimage, coordinate_ls

def testrun():
    import time
    start = time.time()
    pil_image = Image.open("./example/test.jpg")
    detectimagels, classnamels, drawimage, coordinate_ls = yolo_data.detect_image(pil_image)
    end_time = time.time() - start
    print(detectimagels)
    print(classnamels)
    print(drawimage)
    print(coordinate_ls)
    print("推論時間：", str(round(end_time, 3)), "秒")
    drawcv2image = pil2cv(drawimage)
    return drawcv2image


yolo_data = YOLO()
if len(sys.argv) >= 2:
    sys_param1 = str(sys.argv[1])
    if sys_param1 == "test":
        drawcv2image = testrun()
        if len(sys.argv) == 3:
            sys_param2 = str(sys.argv[2])
            if sys_param2 == "show":
                imshow(drawcv2image)
