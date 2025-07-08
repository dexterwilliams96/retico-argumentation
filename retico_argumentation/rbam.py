import retico_core
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class ArgumentRelationIU(retico_core.IncrementalUnit):
    """An Incremental Unit capturing an assumed argumentative relation between utterances."""

    @staticmethod
    def type():
        return "Argument Relation Incremental Unit"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 payload=None, decision=None, source=None, target=None, relation=None, **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=payload, decision=decision, **kwargs)
        self.source = source
        self.target = target
        self.relation = relation

    def get_source(self):
        return self.source

    def set_source(self, source):
        self.source = source

    def get_target(self):
        return self.target

    def set_target(self, target):
        self.target = target

    def get_relation(self):
        return self.relation

    def set_relation(self, relation):
        self.relation = relation

class RbAMModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "RbAM Module"

    @staticmethod
    def description():
        return "A module that performs relation-based argument mining on utterances."

    @staticmethod
    def input_ius():
        return [UtteranceIU]

    @staticmethod
    def output_iu():
        return ArgumentRelationIU

    def __init__(self, model_id="raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L", retroactive_relations=False, device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu",**kwargs):
        super().__init__(**kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
        self.retroactive_relations=False
        self.utterances = dict()

    def process_update(self, update_message):
        um = retico_core.UpdateMessage()
        # Maintain a list of utterance ius, with commit status (true/false)
        # Maintain a list of removed ius
        # Maintain a list of added ius
        if len(um) > 0:
            self.append(um)

