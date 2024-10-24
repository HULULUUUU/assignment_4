import jetson.inference
import jetson.utils
import numpy as np
import argparse
import sys
import time
import cv2

parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

def calculate_angle(id1, id2, id3):
    point1_x = np.array(pose.Keypoints[id1].x)
    point1_y = np.array(pose.Keypoints[id1].y)
    point2_x = np.array(pose.Keypoints[id2].x)
    point2_y = np.array(pose.Keypoints[id2].y)
    point3_x = np.array(pose.Keypoints[id3].x)
    point3_y = np.array(pose.Keypoints[id3].y)
    v1 = (point1_x - point2_x, point1_y - point2_y)
    v2 = (point3_x - point2_x, point3_y - point2_y)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

while True:

    img = input.Capture()

    img_cv = jetson.utils.cudaToNumpy(img)
    
    poses = net.Process(img, overlay=opt.overlay)

    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print("------------------------------------------------------------------------------")
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)
        
        idx_1 = pose.FindKeypoint('left_shoulder')
        idx_2 = pose.FindKeypoint('left_elbow') 
        idx_3 = pose.FindKeypoint('left_wrist')
        if idx_1 < 0 or idx_2 < 0 or idx_3 < 0:
            continue
        angle_left_arm = calculate_angle(idx_1, idx_2, idx_3)
        

        idx_1 = pose.FindKeypoint('right_hip')
        idx_2 = pose.FindKeypoint('right_knee') 
        idx_3 = pose.FindKeypoint('right_ankle')
        if idx_1 < 0 or idx_2 < 0 or idx_3 < 0:
            continue
        angle_right_leg = calculate_angle(idx_1, idx_2, idx_3)
        

        idx_1 = pose.FindKeypoint('left_hip')
        idx_2 = pose.FindKeypoint('left_knee') 
        idx_3 = pose.FindKeypoint('left_ankle')
        if idx_1 < 0 or idx_2 < 0 or idx_3 < 0:
            continue
        angle_left_leg = calculate_angle(idx_1, idx_2, idx_3)

        print(f"Angle of the left arm is: {angle_left_arm:.2f}")
        print(f"Angle of the left leg is: {angle_left_leg:.2f}")
        print(f"Angle of the right leg is: {angle_right_leg:.2f}")
        print("------------------------------------------------------------------------------")
        

        cv2.putText(img_cv, f"Left Arm: {angle_left_arm:.2f} degrees", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img_cv, f"Right Leg: {angle_right_leg:.2f} degrees", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img_cv, f"Left Leg: {angle_left_leg:.2f} degrees", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        

        timestamp = int(time.time() * 1000) 
        filename = f"pose_estimation_{timestamp}.jpg"
        cv2.imwrite(filename, img_cv)

    output.Render(img)

    output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    net.PrintProfilerTimes()

    if not input.IsStreaming() or not output.IsStreaming():
        break