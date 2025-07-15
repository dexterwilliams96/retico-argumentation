import os
import sys

os.environ['WHISPER'] = 'retico-whisperasr'
sys.path.append(os.environ['WHISPER'])

from retico_core import *
from retico_core.audio import MicrophoneModule
from retico_speakerdiarization.utterance import UtteranceModule
from retico_speakerdiarization import SpeakerDiarizationModule
from retico_argumentation.afsolver import AFModule
from retico_argumentation.rstparser import RSTModule
from retico_argumentation.rbam import RbAMModule
from retico_whisperasr.whisperasr import WhisperASRModule


mic = MicrophoneModule()
rbam = RbAMModule()
sd = SpeakerDiarizationModule(
    audio_path='audio', sceptical_threshold=0.4)
asr = WhisperASRModule()
um = UtteranceModule()
rst = RSTModule()
af = AFModule()

mic.subscribe(asr)
mic.subscribe(sd)
sd.subscribe(um)
asr.subscribe(um)
um.subscribe(rst)
um.subscribe(rbam)
rbam.subscribe(af)
rst.subscribe(af)

mic.run()
sd.run()
asr.run()
um.run()
rst.run()
rbam.run()
af.run()

input('Press enter to output AF')

af.stop()
rbam.stop()
rst.stop()
um.stop()
asr.stop()
sd.stop()
mic.stop()
