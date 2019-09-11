# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""
import cognitive_face as CF
import numpy as np
import tensorflow as tf
import cv2
import handsegment as hs
import time
import rnn_eval as re
import predict_spatial as ps
import pickle
import tflearn
import sys
from rnn_utils import get_network_wide, get_data

imagePath = 'test.jpg'
modelFullPath = './output_graph.pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = './output_labels.txt'                                   # 읽어들일 labels 파일 경로

# Replace with a valid subscription key (keeping the quotes in place).
KEY = '9b3e66f7f6db4361b2e1073ed554a5f9'
CF.Key.set(KEY)

# Replace with your regional Base URL
BASE_URL = 'https://faceapi125.cognitiveservices.azure.com/face/v1.0'
CF.BaseUrl.set(BASE_URL)
cntEmo = np.zeros(8, dtype=int)
facelist = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
signlabels = re.load_labels('retrained_labels.txt')





def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()

    img_url = 'face.jpg'
    face_emo = False

    Started = False
    countframe = 0
    numframe = 0
    net = get_network_wide(25, 2048, 29)
    model = tflearn.DNN(net, tensorboard_verbose=0)

    try:
        model.load('checkpoints/' + 'pool.model')
        print("\nModel Exists! Loading it")
        print("Model Loaded")
    except Exception:
        print("\nNo previous checkpoints of %s exist" % ('pool.model'))
        print("Exiting..")
        sys.exit()


    with tf.Session() as sess:
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        cap = cv2.VideoCapture(0)

        while True:
            start = time.time()
            ret, frame = cap.read()
            oriframe = frame
            frame = hs.handsegment(frame)
            if countframe % 5 == 0:

                cv2.imwrite('./test.jpg',frame)
                image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
                predictions = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data })
                predictions = np.squeeze(predictions)

                top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
                for node_id in top_k:
                    human_string = labels[node_id]
                    score = predictions[node_id]
                    #print('%s (score = %.5f)' % (human_string, score))
                answer = labels[top_k[0]]
                print("ans : " + str(answer))
            #print("time : ", time.time()-start)

            if Started != True and str(answer) == 'b\'start\\n\'':
                Started = True

            if Started == True and str(answer) == 'b\'finish\\n\'':
                Started = False
                countframe = 0
                numframe = 0

            if Started == True:
                if countframe % 2 ==0:
                    cv2.imwrite('frames/candy/test' + str(int(countframe / 2)) + '.jpg', frame)

                if countframe != 0 and countframe % 4 == 0:
                    dst = cv2.resize(oriframe, dsize=(160, 120), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('face.jpg', dst)
                    faces = CF.face.detect(img_url, face_id=False, attributes='emotion')
                    print(faces)
                    maxEmo = 0
                    maxVal = 0

                    for face in faces:
                        emotion = face['faceAttributes']['emotion']
                        i = 0
                        for emo in face['faceAttributes']['emotion']:
                            if face_emo == False:
                                if i == 7:
                                    face_emo = True
                                facelist[i] = emo

                            val = emotion[emo]
                            if maxVal < val:
                                maxEmo = i
                                maxVal = val
                            i += 1
                        cntEmo[maxEmo] += 1
                        maxVal = 0
                    #print(countframe)

                if numframe > 5:
                    countframe += 1

                numframe += 1

            if Started == True and countframe == 50:
                #print(facelist)
                maxVal = 0
                res = 'anger'
                i = 0
                for k in cntEmo:
                    if maxVal < k:
                        maxVal = k
                        res = facelist[i]
                    i += 1
                print(res)
                predictions = ps.predict_on_frames('frames', 'retrained_graph.pb', 'Placeholder',
                                                   "module_apply_default/InceptionV3/Logits/GlobalPool", 25)
                out_file = 'predicted-frames-test.pkl'

                with open(out_file, 'wb') as fout:
                    pickle.dump(predictions, fout)
                re.eval_video(out_file, 25, countframe, signlabels, model)
                countframe = 0

if __name__ == '__main__':
    run_inference_on_image()
