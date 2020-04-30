#keras-yolov3モデルを推論させるテンプレ

```python
from preduct_api import detectrun

#返り値
detectimagels, classnamels, drawimage, coordinate_ls = detectrun(pil_image)
```

###detectimagels
[<PIL.Image.Image image mode=RGB size=885x948 at 0x20C160535C0>, <PIL.Image.Image image mode=RGB size=916x819 at 0x20C16053710>]

###classnamels
['label1', 'label2']

###drawimage
<PIL.Image.Image image mode=RGB size=4032x3024 at 0x20C15FAFC50>

###coordinate_ls
[{'label': 'meter', 'xmin': 2077, 'ymin': 895, 'xmax': 2962, 'ymax': 1843}, {'label': 'meter', 'xmin': 1011, 'ymin': 939, 'xmax': 1927, 'ymax': 1758}]



#使い方

```shell script
git clone https://github.com/foasho/yolov3-prediction.git
```

##1. predictionフォルダの中にtarget_model.h5ファイルを入れる

##2. class_write.pyを実行して、ラベル名を生成

```shell script
python class_write.py label1 label2 label3 ...
```

##3. デモ方法：exampleフォルダの中test.jpgを入れて実行する

```shell script
#画像結果を表示させる場合
python predict_api.py test show

#コンソール画面にのみ表示させる場合
python predict_api.py test
```



