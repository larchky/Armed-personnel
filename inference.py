from detection.armed_detection import ArmedPersonDetector

detector = ArmedPersonDetector(
    pose_model_path="path/to/pose.pt",
    gun_model_path="path/to/gun.pt",
    video_path="input.mp4",
    save_path="output.mp4",
    conf=0.7
)
detector.run()
