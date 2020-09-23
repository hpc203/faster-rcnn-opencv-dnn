训练模型的下载地址是 http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
下载完成后解压到faster_rcnn_inception_v2_coco_2018_01_28文件夹里，graph.pbtxt已经在里面的，不需要运行程序生成它。
关于graph.pbtxt的生成方法，我在此强调一下。首先下载opencv(地址是 https://github.com/opencv/opencv)到本地
然后cd到opencv文件夹里的samples/dnn，这是可以看到里面有tf_text_graph_faster_rcnn.py，在终端运行 python tf_text_graph_ssd.py --input=/path/to/model.pb --config=/path/to/example.config --output=/path/to/graph.pbtxt
注意路径要写对，运行程序后就可以得到graph.pbtxt
准备就绪后，运行python main_detect_img.py
