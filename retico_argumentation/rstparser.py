import retico_core
import torch

from isanlp_rst.parser import Parser
from retico_argumentation.utterance import UtteranceIU

class RSTIU(UtteranceIU):
    """An Incremental Unit capturing the RST structure of an utterance."""

    @staticmethod
    def type():
        return "Utterance Incremental Unit"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 payload=None, decision=None, speaker=None, text=None, tree=None, **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=payload, decision=decision, speaker=speaker, text=text, **kwargs)
        self.tree = tree

    def get_tree(self):
        return self.tree

    def set_text(self, tree):
        self.tree = tree

class RSTModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "RST Module"

    @staticmethod
    def description():
        return "A module that extracts an RST tree from an utterance."

    @staticmethod
    def input_ius():
        return [UtteranceIU]

    @staticmethod
    def output_iu():
        return RSTIU

    def __init__(self, device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu", **kwargs):
        super().__init__(**kwargs)
        self.parser = Parser(hf_model_name="tchewik/isanlp_rst_v3", hf_model_version="gumrrg", cuda_device=device)

    def process_update(self, update_message):
        um = retico_core.UpdateMessage()
        # We may want to join utterances, but we'll reflect on this TODO
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.REVOKE:
                remove_iu = None
                for curr_iu in self.current_output:
                    if curr_iu.grounded_in == iu:
                        remove_iu = curr_iu
                        break
                if remove_iu is not None:
                    self.revoke(remove_iu)
                    um.add_iu(remove_iu, ut)
            elif ut == retico_core.UpdateType.ADD:
                tree = self.parser(iu.get_text())["rst"]
                tree[0].to_rs3('filename.rs3')
                print('dumped!')
            else:
                tree = self.parser(iu.get_text())["rst"]
                tree[0].to_rs3('filename.rs3')
                print('dumped!')
        # If revoke, revoke iu if it exists with that utterance as grounded
        # If add, parse tree, and add
        # If commit, check if exists and commit if does, otherwise create new message and commit by parsing
        if len(um) > 0:
            self.append(um)
