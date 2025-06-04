#Class for clip and record the training data
from moviepy import ImageSequenceClip
import os
import glfw
from pyvirtualdisplay.smartdisplay import SmartDisplay
display = SmartDisplay(visible=0, size=(1920,1080),fbdir='/tmp')
display.start()
glfw.init()
available_fbconfigs = glfw.get_video_modes(glfw.get_primary_monitor())
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'


class Recorder():
    def __init__(self, gameName):
        self.gameName=gameName
        self.basePath="./clips/"+gameName
        self.pathTraining=self.basePath+"/training/"
        self.pathEvaluation=self.basePath+"/evaluation/"
    

    def saveRecord(self, frames,episodeNumber=0,evaluation=False):
        if evaluation:
            path=self.pathEvaluation
        else:
            path=self.pathTraining
        if not os.path.exists(path):
            os.makedirs(path)
        clip = ImageSequenceClip(frames, fps=15)
        clipName=path+self.gameName+"_"+str(episodeNumber)+".mp4"
        clip.write_videofile(clipName,logger=None)
        clip.close()