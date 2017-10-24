# -*- coding: GB18030 -*-
#wujingwei
#useage: python runmain.py --queryFile  dataset/xxx.png --indexFile index.csv --result_path result
import csv,os
import argparse
import cv2
from picture_descriptor.ClolorDescriptor import ColorDescriptor
from picture_descriptor.searcher import Searcher
aparse = argparse.ArgumentParser()
aparse.add_argument("-q", "--queryFile", required=True, help="Path to the query image")
aparse.add_argument("-i", "--indexFile", required=True, help="Path to where the computed index will be stored")
aparse.add_argument("-r", "--result_path", required=True, help = "Path to the result path")
args = vars(aparse.parse_args())

cd = ColorDescriptor((8, 12, 3)) #��ʼ��ͼƬ����
query_img = cv2.imread(args["queryFile"])
features = cd.describe(query_img) #��ȡָ��ͼƬ������

searcher = Searcher(args["indexFile"])
results = searcher.search(features)
print results

#cv2.imshow("Query", query_img) #��ʾָ����ͼƬ

for (score, resultID) in results:
	print score,resultID
	# ��ʾ�������ķ���Ҫ�����Ƭ
	result = cv2.imread(args["result_path"] + "/" + resultID)
	#print result
	#cv2.imshow("Result", result)
	#cv2.waitKey(0)