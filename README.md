Pothole detection system

Required Libraries :

flask
opencv-python
ultralytics
pydantic
pymongo
requests
numpy

To install run 
    pip install -r requirement.txt 

Website:
    Index page has three option 
        Image: upload image to detect pothole ,on upload give alert about pothole
                if pothole detected mark the pothole
        Video: upload the video, on upload result video will generated 
                can take some time
                result has marked pothole with a highlight about road condition
        Webcam: live pothole prediction and if detected coordinates along with image is stored in database
                location is ip based (for visualisation not actual use)
                to stop webcam click on stop button
