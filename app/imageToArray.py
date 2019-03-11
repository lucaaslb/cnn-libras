from PIL import Image
from random import shuffle

def imgToArray(img):

	img = img.resize((28, 28), Image.ANTIALIAS)

	pixel = img.load()

  #Should be 28, 28 as img.resize above
	width, height = img.size

	arq = open('img.txt', 'w')
	arrayImg = []
	for x in range(0,width):
		for y in range(0,height):
			arrayImg.append(pixel[y,x])   
			arq.write(str(pixel[y,x])+',')

	return arrayImg