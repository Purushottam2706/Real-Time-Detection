Download all the module from requirement.txt file and directly run the main file
<!-- pip install -r requirement.txt -->
go in the main.py file 
at last there are 
# detect(0) ---> for real time camera detection
# detect("VIDEOS/bikes.mp4") --> a bike video object detection
# detect("VIDEOS/people.mp4") --> a video of people it will detect all of them
# detect("VIDEOS/motorbikes-1.mp4") --> it is for detecting moter car in a very large video 
# tracker() --> it will count the numbe rof cars in the video

at the top 
 
model = yolo("../Yolo-Weights/yolov8n.pt") --> yolov8n "n -> means nano" this is the nano version of yolo 
you can also use the large version just replace n with l 
like this --> model = yolo("../Yolo-Weights/yolov8l.pt") 


sort.py is just a frame sorting algorithm 
this is half git  repo of the project the next half i have not uploaded  yet 
that half detect the number plate or licence plate of the car and record the data into csv file form like care licence plate 
