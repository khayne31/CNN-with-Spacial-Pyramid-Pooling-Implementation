import os, requests
from bs4 import BeautifulSoup
from PIL import Image
#import urllib2
import json
import urllib
import sys
import urllib3
import time

cwd = os.getcwd()
final_directory = os.path.join(cwd, r"images")
if not os.path.exists(final_directory):
	os.makedirs(final_directory)

dir_names = ["People", "Non People", "Dogs"]

for item in dir_names:
	new_directory = os.path.join(cwd+"\images", item)
	if not os.path.exists(new_directory):
		os.makedirs(new_directory)

#seperates individual urls and adds it to a new file
f = open("urls.txt", "r")
l = open("newurls.txt", "a+")
#print(f.readlines())
urls = []
for line in f.readlines():
	#print(line)
	if "https" in line:
		#print(type(line))
		l.write(line)
		if ".jpg" in line:
			urls.append(line.partition(".jpg")[0]+".jpg")
		elif ".png" in line:
			urls.append(line.partition(".png"[0]+".png"))

#gets the image based on the url and downloads it
#TODO: parse the flile and download every image associated with every url
def download_images():
	for i in range(len(urls)):
		try:
			get = requests.get(urls[i], timeout = None)
			img_data = get.content
			print(get.status_code)
			sys.stdout.flush()
			if get.status_code == 200:
				if ".jpg" in urls[i]:
					with open("images/Dogs/dog"+str(i) + ".jpg", "wb") as handler:
						handler.write(img_data)
				elif ".png" in urls[i]:
					with open("images/Dogs/dog"+str(i) + ".png", "wb") as handler:
						handler.write(img_data)
		except urllib.error.HTTPError as e:
			print("1Error code", e.code)
		except urllib.error.URLError as e:
			print("1Reason: ", e.reason)
		except urllib3.exceptions.ProtocolError as e:
			print(e.code)
		except requests.exceptions.ConnectionError as e:
			print("connection request error")
		except requests.exceptions.InvalidSchema as e:
			print("Invalid Schema")
		except http.client.RemoteDisconnected as rd:
			print("Remote disconnected error")

#opens the file
#os.startfile("image.jpg")

#this gets the pixel valies for the image
#TODO: find a way to store the pixel values associated with every url for later use by the convnet

dirs = cwd+"\images\Dogs"
files = [f for f in os.listdir(dirs)]
lengths = []
print(len(files))
for i in range(len(files)):
	img = Image.open("images/Dogs/"+files[i])
	lengths.append((len(list(img.getdata())),files[i]))
print(sorted(lengths, reverse  = True)[0])
"""get = requests.get("https://upload.wikimedia.org/wikipedia/commons/d/d9/Collage_of_Nine_Dogs.jpg ")
get2 = requests.get(urls[2].partition(".jpg")[0]+".jpg")
print(get2.status_code)
print("https://upload.wikimedia.org/wikipedia/commons/d/d9/Collage_of_Nine_Dogs.jpg" == urls[2])
print(urls[2][-1])
print("https://upload.wikimedia.org/wikipedia/commons/d/d9/Collage_of_Nine_Dogs.jpg")"""
