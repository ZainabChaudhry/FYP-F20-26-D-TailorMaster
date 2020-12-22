# FYP-F20-26-D-TailorMaster
An android application that will help you get your customized design and anthropometric measurements from a 2D video of the user
# Iteration 2:
you need to run the all the tabs of the jupyter notebook.
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
The graph_opt.pb file is used for object detection and getting reference points of these body parts. TheposeDetector functions helps track the reference points and The getMeasurementsInPixels helps obtain the distance bertween points detected and ifnally the getOriginalMeasurements function helps get measurements in inches
