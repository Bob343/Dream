#1.1
import urllib2
import sys
import webbrowser
sys.path.append("lib-tk")
sys.path.append("caffe-master/python")
sys.path.append("cv2.so")
sys.path.append("io.pyc")
sys.path.append("numpy")
sys.path.append("scipy/ndimage")
sys.path.append("skimage")
sys.path.append("spatial")
sys.path.append("spatial/qhull.so")
sys.path.append("PIL")
sys.path.append("protobuf-2.6.1-py2.7.egg")
from google.protobuf import text_format
import Tkinter
import os
import cv2
from io import StringIO
import numpy as np
import scipy.ndimage as nd
from PIL import  ImageTk
import PIL.Image
import caffe

def showarray(a, fmt="JPEG"):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    out = PIL.Image.fromarray(a)
    l.place_forget()
    status.place_forget()
    img = ImageTk.PhotoImage(out)
    imageL.configure(image = img,anchor = 'n')
    imageL.image = img
    imageL.jpeg = out
model_path = '/root/Desktop/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
file1 = open(net_fn)
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(file1.read(), model)
file1.close()
model.force_backward = True
tmp = open('tmp.prototxt', 'w')
tmp.write(str(model))
tmp.close()
net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), 
                       channel_swap = (2,1,1))
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    sys.stdout.flush()
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
    net.forward(end=end)
    dst.diff[:] = dst.data 
    net.backward(start=end)
    g = src.diff[0]
    src.data[:] += step_size/np.abs(g).mean() * g
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)   
def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    j = 1
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, .7142,.7142), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        
	if octave > 0:
            # upsca`le details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

       	src.reshape(1,3,h,w) # resize the network's input image size
       	src.data[0] = octave_base+detail
        for i in xrange(iter_n):
	    lab.set("Iteration: "+str(j))
	    stat.set("Processing")
	    top.update_idletasks()
	    j +=1
	   # if cv2.waitKey(1) & 0xFF == ord('q'):
	#	break
            make_step(net, end=end, clip=clip, **step_params)
	    vis = deprocess(net,src.data[0])
	    if not clip:
		vis = vis*(255.0/np.percentile(vis,99.98))
	    showarray(vis)
	    width,height = imageL.jpeg.size
	    l.place(x = 398, y = height+10)
 	    status.place(x = 502, y = height+10)
            imageL.place(x = (450-(width/2)), y = 0)
	    top.update_idletasks()
        # extract details produced on the current octave
       	detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])
def func1():
	splt = "split"
	items = dreamList.curselection()
	if  splt in str(net.blobs.keys()[items[0]+1]):
		stat.set("Bad filter, select another.")
		status.config(fg = 'red')
		top.update_idletasks()
	elif len(items) > 0:
		stat.set("Processing")
		status.config(fg = "red")
		width,height = imageL.jpeg.size
		status.place(x = 502,y = height+10)
		top.update_idletasks()
		img = np.float32(imageL.jpeg)
		for i in range(int(cycles.get())):
			img = deepdream(net, img,end = net.blobs.keys()[items[0]+1])
		img = PIL.Image.fromarray(np.uint8(img))
		stat.set("Done!")
		status.config(fg = "green")
		top.update_idletasks()
	else:
		stat.set("Select a filter!")
		status.config(fg = "red")
		width,height = imageL.jpeg.size
		status.place(x = 502, y = height+10)
		top.update_idletasks()
	#img.save("/root/Desktop/j.jpg")
def askFile():
	name = openName.get()
	if(os.path.exists(name)):
		step = PIL.Image.open(name)
		img = ImageTk.PhotoImage(step)
		imageL.configure(image = img)
		imageL.image = img
		imageL.jpeg = step
	        width,height = imageL.jpeg.size
	    	l.place(x = 398, y = height + 10)
		status.place(x = 502, y = height+10)
def capWindow():
	imageL.going = True
	cam = cv2.VideoCapture(int(camSelection.get()))
	while(imageL.going):
		ret, frame = cam.read()
		#top.bind('<Key>',notGoing)
		#imageL.bind('<Key>',notGoing)
		cv2.imshow("Capture",frame)
		#cv2.imwrite(file,frame)
		#inimg = ImageTk.PhotoImage(PIL.Image.open("/root/temp.jpg"))
		#imageL.configure(image = inimg)
		#imageL.focus_set()
		#top.update_idletasks()
		if cv2.waitKey(1) & 0xFF == ord('q'):
		#if ord(z) == ord('q'):
			imageL.going = False
	retval,im = cam.read()
	test = None 
	cam.release()
	del(cam)
	cv2.destroyWindow("Capture")
	cv2.imshow("Capture",im)
	file = "/root/temp.jpg"
	cv2.imwrite(file,im)
	inimg = PIL.Image.open("/root/temp.jpg")
	imageL.jpeg = inimg
	width,height = inimg.size
	l.place(x = 398, y = height+10)
	status.place(x = 502, y = height+10)
	#trans.resize(inimg.size)
	#if(os.path.exists(logoText.get())):
		#trans = PIL.Image.open(logoText.get()).resize((200,200)) #Input superimpose dimensions here <<<<
		#inimg.paste(trans,((int(inimg.size[0]) -int( trans.size[0])),(int(inimg.size[1]) - int(trans.size[1]))),trans)
	img = ImageTk.PhotoImage(inimg)
	imageL.configure(image = img)
	imageL.image = img
	os.remove("/root/temp.jpg")
def saveDream():
	stat.set("Saving!")
	status.config(fg = "green")
	top.update_idletasks()
	if(os.path.exists("/root/Pictures/Logos/"+logoText.get()) and len(logoText.get()) != 0):
		trans = PIL.Image.open("/root/Pictures/Logos/"+logoText.get()).resize((150,150)) #Input superimpose dimensions here <<<<
		imageL.jpeg.paste(trans,((int(imageL.jpeg.size[0]) -int( trans.size[0])),(int(imageL.jpeg.size[1]) - int(trans.size[1]))),trans)
	imageL.jpeg.save("/root/Pictures/"+savingName.get()+".jpeg")
imageI = 0
def changed(a,b,c):
	if(os.path.isfile("/root/Pictures/Logos/"+logoText.get())):
		logoFileText.set("Logo found!")
		
	else:
		logoFileText.set("No logo detected!")
	top.update_idletasks()
def saveChanged(a,b,c):
	if(os.path.isfile("/root/Pictures/"+savingName.get()+".jpeg")):
		checkSaveName.set("File exists! Saving will overwrite.")
	else:
		checkSaveName.set("File does not exist.")
		
def openChanged(a,c,b):
	if(os.path.isfile(openName.get())):
		checkOpenName.set("File found! Click open.")
	else:
		checkOpenName.set("File not found.")
cams = []
for i in range(10):
	testCam = cv2.VideoCapture(i)
	if(testCam.isOpened()):
		cams.append(i)
	del(testCam)
if not os.path.exists("/root/Pictures/Logos"):
	os.makedirs("/root/Pictures/Logos")
	

name = ""
top = Tkinter.Tk()
top.geometry('900x730')
top.title('Google Dream')
lab = Tkinter.StringVar()
lab.set("Iteration: "+str(0))
l = Tkinter.Label(top,textvariable = lab)
l.place(x = 450, y = 480)


B = Tkinter.Button(top, text = "Start", command = func1)
B.place(x = 665, y= 630)
cycles = Tkinter.StringVar()
cycles.set("1")
cyclesEntry = Tkinter.Entry(top,width = 3, textvariable = cycles)
cyclesEntry.place(x = 730, y = 635)


capture = Tkinter.Button(top,text = "Capture Image", command = capWindow)
capture.place(x = 665, y = 590)
imageL = Tkinter.Label(top,anchor = 'center')
imageL.place(x = 130, y = 0)

logoText = Tkinter.StringVar()
logo = Tkinter.Entry(top, width = 25,textvariable = logoText)
logo.place(x = 400, y = 635)
logoText.trace('w',changed)
logoFileText = Tkinter.StringVar()
logoFileText.set("No logo detected!")
logoFile = Tkinter.Label(top,textvariable = logoFileText)
logoFile.place(x = 400, y = 660)
logLabel = Tkinter.Label(top, text = "Logo:")
logLabel.place(x = 360, y = 635)


checkSaveName = Tkinter.StringVar()
checkSaveName.set("File does not exist.")
checkSave = Tkinter.Label(top, textvariable = checkSaveName)
checkSave.place(x = 400, y = 575)
savingName = Tkinter.StringVar()
saveName = Tkinter.Entry(top, width = 25,textvariable = savingName)
savingName.trace('w',saveChanged)
saveName.place(x=400,y=555)
saveLabel = Tkinter.Label(top,text = "Save Name:")
saveLabel.place(x = 320, y = 555)
save = Tkinter.Button(top,text = "Save", command = saveDream).place(x = 584, y = 550)


openLabel = Tkinter.Label(top,text = "Open Name:")
openLabel.place(x = 320, y = 595)
openingName = Tkinter.StringVar()
checkOpenName = Tkinter.StringVar()
checkOpenName.set("File does not exist.")
checkOpen = Tkinter.Label(top, textvariable = checkOpenName)
checkOpen.place(x=400, y = 615)
openName = Tkinter.Entry(top, width = 25,textvariable = openingName)
openName.place(x=400, y = 595)
openingName.trace('w',openChanged)
browse = Tkinter.Button(top, text = "Open", command = askFile).place(x = 584, y = 590)

def callback(event):
	webbrowser.open_new("https://github.com/Bob343/Dream/blob/master/dream.py")

currentVersion = 1.1
versionString = Tkinter.StringVar()
versionString.set("Version: "+str(currentVersion))
line = urllib2.urlopen("https://raw.githubusercontent.com/Bob343/Dream/master/dream.py").read().split("\n")
line = line[0].replace("#","")
line = line.replace("\n","")
versionLabel = Tkinter.Label(top,textvariable =versionString)
versionLabel.place(x=800, y = 5)
newVersion = Tkinter.Label(top,text = "New Version!!!",fg = "blue",cursor = "hand2")
newVersion.bind("<Button-1>",callback)
if(currentVersion != float(line)):
	newVersion.place(x = 800, y =25)

stat = Tkinter.StringVar()
status = Tkinter.Label(top,textvariable = stat)
status.place(x = 0, y = 0)
list1 = net.blobs.keys()[1:]
dreamList = Tkinter.Listbox(top,selectmode = "single",width = 30)
for item in list1:
	dreamList.insert("end",item)
filterLabel = Tkinter.Label(top,text = "Filters:")
filterLabel.place(x = 70, y = 535)
dreamList.place(x = 70, y = 555)
imageL.going = True

camSelection = Tkinter.StringVar()
camSelection.set("0")
camOption = Tkinter.OptionMenu(top,camSelection,*cams)
camOption.place(x=720,y = 550)
cameraLabel = Tkinter.Label(top,text = "Camera:")
cameraLabel.place(x = 665, y = 555)
top.mainloop()
