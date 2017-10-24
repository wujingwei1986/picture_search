# -*- coding: GB18030 -*-
#wujingwei
import numpy as np
import cv2

class ColorDescriptor:
	def __init__(self, bins):
		# �Զ���ֱ��ͼ��bin����
		self.bins = bins

	def describe(self, image):
		# ��ͼƬ��BGRת��ΪHSV��ʽ
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		# ����ͼƬ����
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))

		#��ͼƬ�ָ�Ϊ4���֣������ϣ����ϣ����£����£�
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]

		# construct an elliptical mask representing the center of the
		# image
		(axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
		whole_image = np.zeros(image.shape[:2], dtype = "uint8")
		ellipse_shape = cv2.ellipse(whole_image, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		# ѭ����ʾ������ͼ��
		for (startX, endX, startY, endY) in segments:
			whole_image = np.zeros(image.shape[:2], dtype = "uint8")
			rectangle_shape = cv2.rectangle(whole_image, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(rectangle_shape, ellipse_shape) #���μ�ȥ����Բ�غϲ���

			hist = self.histogram(image, cornerMask) #����ÿһ��ͼ�������
			features.extend(hist)

		hist = self.histogram(image, ellipse_shape)
		features.extend(hist)

		return features

	def histogram(self, image, mask):
		# extract a 3D color histogram from the masked region of the
		# image, using the supplied number of bins per channel; then
		# normalize the histogram
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256])
		hist = cv2.normalize(hist,hist,0, 1, cv2.NORM_MINMAX).flatten() #ֱ��ͼԭ�ع�һ��������ά����ת����һά

		return hist
