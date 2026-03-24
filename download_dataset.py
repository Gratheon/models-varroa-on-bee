from roboflow import Roboflow
rf = Roboflow()
project = rf.workspace("varroa-j8231").project("varroa8k")
version = project.version(1)
dataset = version.download("yolov8")
