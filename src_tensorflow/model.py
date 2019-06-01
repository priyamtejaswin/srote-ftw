import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model
from motion import MotionCompensation 
from fusion import EarlyFusion


class ENHANCE(Model):
    """
    <https://www.youtube.com/watch?v=Vxq9yj2pVWk>
    One step closer to the future... unless of course, the eigen value is off. /s
    """

    def __init__(self):
        super(ENHANCE, self).__init__() 

        self.motion_compensate = MotionCompensation() 
        self.earlyfusion = EarlyFusion(timeframes=3, iheight=32, iwidth=32, channels=3)
        
        # conv layers after compensation
        self.conv1 = Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")
        self.conv2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")
        self.conv3 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")
        self.conv4 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")
        self.conv5 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")
        self.conv6 = Conv2D(filters=27, kernel_size=(3,3), strides=(1,1), activation="linear", padding="same")

    def call(self, frames):
        
        # frames is (batchsize x 3 x 32 x 32 x 3)
        comp1, flow1 = self.motion_compensate(frames[:,1,:,:,:], frames[:,0,:,:,:])
        comp2, flow2 = self.motion_compensate(frames[:,1,:,:,:], frames[:,2,:,:,:])

        all_frames = tf.stack((comp1, frames[:,1,:,:], comp2), axis=1)
        ef = self.earlyfusion(all_frames)
        
        out = self.conv1(ef) 
        out = self.conv2(out)
        out = self.conv3(out) 
        out = self.conv4(out) 
        out = self.conv5(out) 
        out = self.conv6(out) 
        upped = tf.nn.depth_to_space(out, 3)

        return upped, (flow1, flow2), (comp1, comp2)

class Mocomp_test(Model):
    """
    This model just chesk if motion compensation is working 
    """ 
    def __init__(self):
        super(Mocomp_test, self).__init__() 
        self.motion_compensate = MotionCompensation() 
    
    def call(self, frames): 
        
        # frames is (batchsize x 2 x 32 x 32 x 3) Why 2 ? because we just want it to predict the optical flow 
        # Note: frame0 is reference, frame1 is "other" frame
        comp1, flow1 = self.motion_compensate(frames[:,0,:,:,:], frames[:,1,:,:,:])

        return comp1, flow1


if __name__ == "__main__":

    print 'Testing...'
    tf.enable_eager_execution()
    print "TF Executing Eagerly?", tf.executing_eagerly()

    x = tf.constant(np.random.rand(5, 3, 32, 32, 3).astype(np.float32))
    enhance = ENHANCE() 
    y = enhance(x)[0]
    print(y.shape)




        
        





