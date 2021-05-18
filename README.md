# FYP-F20-26-D-TailorMaster
An android application that will help you get your customized design and anthropometric measurements from a 2D video of the user
# Iteration 1:
This is the front end of the application that will further be improved( in working) . 
In it user can enter manual measurements if he/she wants (EnterManual Activity) and then get their customized design in 3 ways(DressDesign Activity) . 
1. They can upload an image (Upload image activity)
2. They can draw a customized design of their shirt and than trouser separately (Draw DrawwDesign and DrawDesign2 activity)
3. They can ask for recommendations ( The model to be trained in next iteration) 
Finally they can view their report of design and measurements in the report.

The button to record and video and get 3d measurements from a 2d video is given in main activity which when pressed asks for height,bodytype (mentioned in iteration 2) and gender and saves the video made made in the way the sample video is made in iteration 2 and then the video is passed to model made in iteration 2 for measurement extraction.

# Iteration 2:
you need to run all the tabs of the jupyter notebook.
# the mymain() function
is the main function that calls above all functions when required.
the mymain function first asks for the height of the user, then asks whether he is slim, normal, healthy or bulky
Slim means that the chest of the person is much larger than his/her waist like 3 or more than 3 inches larger
Normal means that the chest of the person is slightly more than their waist .
Healthy means that the chest of the person is slightly smaller than their waist.
Bulky is someone whose chest is more than their normal height and much less than their waist.

Above mentioned are the 4 categories of Body types which the user will choose on the base of their assumptions.
The last thing required before video is the gender as the body parts division of male are different from female.
The path of video must be given that is saved in memory.
# The video must be recorded by closed fist, straight arms and stop for 2 seconds at every 90 degree starting for the one facing the camera.
The person must come straight in the camera frame wearing an atmost half sleeves tshirt that fits them and a tight pants or leggings so that main body parts e.g.knees, ankle, hipe,eyes,shoulders, wrists etc are visible and easily detectable.
# Code explanation:
The model used is mobileNet that is 7MB at total so that deployment can be easier and takes less space moreover the DNN(Deep neural network ) object detection API is used.
The graph_opt.pb file is used for object detection and getting reference points of these body parts. TheposeDetector functions helps track the reference points and The getMeasurementsInPixels helps obtain the distance bertween points detected and ifnally the getOriginalMeasurements function helps get measurements in inches.

# Iteration 3:
you need to run all the tabs of the jupyter notebook as mentiooned in iteration 2
The application is same as the one in iteration 1, the application is updated and enhanced in user experience and both the model and the application is connected.
You need to download all the code on your PC both the model and the application and enter your PCs IPv4 in the android application code.
