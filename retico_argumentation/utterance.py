import retico_core

from retico_core.text import SpeechRecognitionIU
from retico_speakerdiarization.speaker_diarization import SpeakerIU
from sortedcontainers import SortedDict

class UtteranceIU(retico_core.IncrementalUnit):
    """An Incremental Unit representing an utterance from a speaker."""

    @staticmethod
    def type():
        return "Utterance Incremental Unit"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 payload=None, decision=None, speaker=None, text=None, **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=payload, decision=decision, **kwargs)
        self.speaker = speaker
        self.text = text
        self.payload = payload if payload is not None else {"speaker": self.speaker, "text": self.text}

    def get_speaker(self):
        return self.speaker

    def set_speaker(self, speaker):
        self.speaker = speaker
        self.payload["speaker"] = self.speaker

    def get_text(self):
        return self.text

    def set_text(self, text):
        self.text = text
        self.payload["text"] = self.text

class UtteranceModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Utterance Module"

    @staticmethod
    def description():
        return "A module that connects text to it's speaker."

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU, SpeakerIU]

    @staticmethod
    def output_iu():
        return UtteranceIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store a time-sorted list of utterances
        self.utterances = SortedDict()
        # Store a time-sorted list of speaker identificaitons
        self.speaker_timeline = SortedDict()

    def process_update(self, update_message):
        um = retico_core.UpdateMessage()
        speakers_updated = []
        text_added = False
        # check final
        for i, tup in enumerate(update_message):
            iu, ut = tup
            origin = iu.grounded_in.created_at
            # Only deal with text commits
            if isinstance(iu, SpeechRecognitionIU) and ut == retico_core.UpdateType.COMMIT and i == len(update_message) - 1:
                self.utterances[origin] = iu.predictions[0]
                text_added = True
            elif isinstance(iu, SpeakerIU):
                # Update any speakers
                self.speaker_timeline[origin] = iu
                speakers_updated.append(origin)
        if len(speakers_updated) > 0:
            # Update existing IUs
            remove_set = []
            for iu in self.current_output:
                # Check the original audio in timestamp for the speaker id
                origin = iu.grounded_in.grounded_in.created_at
                if origin in speakers_updated:
                    new_speaker = self.speaker_timeline[origin].get_speaker()
                    new_iu = self.create_iu(iu.grounded_in) if iu.get_speaker() != new_speaker[0] else iu
                    # If the speaker id changed then revoke the IU
                    if new_speaker[0] != iu.get_speaker():
                        um.add_iu(iu, retico_core.UpdateType.REVOKE)
                        remove_set.append(iu)
                    # If speaker confirmed and not unknown commit the IU
                    if new_speaker[0] is not None and new_speaker[1]:
                        um.add_iu(new_iu, retico_core.UpdateType.COMMIT)
                        remove_set.append(new_iu)
                    # If the speaker id is uncomfirmed but known then add the IU
                    elif new_speaker[0] is not None and not new_speaker[1]:
                        um.add_iu(new_iu, retico_core.UpdateType.ADD)
                        self.current_output.append(new_iu)
            self.current_output = [iu for iu in self.current_output if iu not in remove_set]

        # Handle new text/existing text
        if len(speakers_updated) > 0 or text_added:
            print(self.utterances)
            print(self.speaker_timeline)
            # Map utterances to their closest speaker to get new IUs
            delete_utterance = []
            for utt_key in self.utterances.keys():
                speaker_keys = self.speaker_timeline.keys()
                for i, speaker_key in enumerate(speaker_keys):
                    utterance = self.utterances[utt_key]
                    speaker = self.speaker_timeline[speaker_key]
                    # Don't process anything hanging (TODO this will mean the last utterance is not processed)
                    if i != len(speaker_keys) - 1:
                        # Anything before first speaker is first speaker
                        if i == 0 and utt_key < speaker_key:
                            delete_utterance.append(utt_key)
                            speaker_id, confirmed = speaker.get_speaker()
                            if speaker_id is not None:
                                output_iu = self.create_iu(speaker)
                                output_iu.set_speaker(speaker_id)
                                output_iu.set_text(utterance)
                                if confirmed:
                                    um.add_iu(output_iu, retico_core.UpdateType.COMMIT)
                                else:
                                    um.add_iu(output_iu, retico_core.UpdateType.ADD)
                                    self.current_output.append(output_iu)
                        elif i != 0 and utt_key >= speaker_key and utt_key <= speaker_keys[i + 1]:
                            # Check which speaker the utterance is closer to by comparing the grounded audio input
                            prev_dist = utt_key - speaker_key
                            next_key = speaker_keys[i + 1]
                            next_dist = next_key - utt_key
                            if next_dist < prev_dist:
                                speaker = self.speaker_timeline[next_key]
                            delete_utterance.append(utt_key)
                            speaker_id, confirmed = speaker.get_speaker()
                            if speaker_id is not None:
                                output_iu = self.create_iu(speaker)
                                output_iu.set_speaker(speaker_id)
                                output_iu.set_text(utterance)
                                if confirmed:
                                    um.add_iu(output_iu, retico_core.UpdateType.COMMIT)
                                else:
                                    um.add_iu(output_iu, retico_core.UpdateType.ADD)
                                    self.current_output.append(output_iu)
            for key in delete_utterance:
                del self.utterances[key]
        if len(um) > 0:
            self.append(um)
