import retico_core
import torch

from isanlp_rst.parser import Parser
from retico_speakerdiarization.utterance import UtteranceIU


class RSTIU(retico_core.IncrementalUnit):
    """An Incremental Unit capturing the RST structure of an utterance."""

    @staticmethod
    def type():
        return "RST Incremental Unit"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 payload=None, decision=None, tree=None, **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=payload, decision=decision, **kwargs)
        self.tree = tree

    def get_tree(self):
        return self.tree

    def set_tree(self, tree):
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

    def __init__(self, model_id="tchewik/isanlp_rst_v3", model_version="gumrrg", device=torch.cuda.current_device() if torch.cuda.is_available() else -1, **kwargs):
        super().__init__(**kwargs)
        self.parser = Parser(hf_model_name=model_id,
                             hf_model_version=model_version, cuda_device=device)

    def process_update(self, update_message):
        um = retico_core.UpdateMessage()

        for iu, ut in update_message:
            if ut == retico_core.UpdateType.REVOKE:
                remove_iu = None
                for curr_iu in self.current_output:
                    if curr_iu.grounded_in == iu:
                        remove_iu = curr_iu
                        break
                if remove_iu is not None:
                    self.current_output.remove(remove_iu)
                    um.add_iu(remove_iu, ut)
            elif ut == retico_core.UpdateType.ADD:
                tree = self.parser(iu.get_text())["rst"]
                output_iu = self.create_iu(iu)
                output_iu.set_tree(tree)
                um.add_iu(output_iu, ut)
                self.current_output.append(iu)
            else:
                remove_iu = None
                for curr_iu in self.current_output:
                    if curr_iu.grounded_in == iu:
                        found = curr_iu
                        break
                if remove_iu is not None:
                    self.current_output.remove(remove_iu)
                    um.add_iu(remove_iu, ut)
                else:
                    tree = self.parser(iu.get_text())["rst"]
                    output_iu = self.create_iu(iu)
                    output_iu.set_tree(tree)
                    um.add_iu(output_iu, ut)

        if len(um) > 0:
            self.append(um)
