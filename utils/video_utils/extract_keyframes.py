import os

import cv2

import operator
import numpy as np


class Frame:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def _smooth(x, window_len=13, window='hanning'):
    """_smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the _smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average _smoothing.

    output:
        the _smoothed signal

    example:

    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = _smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    """
    if x.ndim != 1:
        raise ValueError("_smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


def _rel_change(a, b):
    x = (b - a) / max(a, b)
    return x


def extract_keyframes(video_path: str, dst_dir: str, n_frames: int,):
    cap = cv2.VideoCapture(video_path)

    prev_frame = None
    
    frame_diffs = []
    frames = []
    ret, frame = cap.read()
    i = 1
    
    while ret:
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            count = np.sum(diff)
            frame_diffs.append(count)
            frame = Frame(i, frame, count)
            frames.append(frame)
        prev_frame = curr_frame
        i = i + 1
        ret, frame = cap.read()

    cap.release()
    
    # sort the list in descending order
    frames.sort(key=operator.attrgetter("value"), reverse=True)
    for keyframe in frames[:n_frames]:
        cv2.imwrite(os.path.join(dst_dir, f'frame_{str(keyframe.id)}.png'),
                    keyframe.frame)
