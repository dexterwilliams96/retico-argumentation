import retico_core
import torch

from retico_speakerdiarization.utterance import UtteranceIU
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig


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
        self.payload = {"relation": relation}

    def __repr__(self):
        return "%s - (%s): %s" % (
            self.type(),
            self.creator.name(),
            str(self.payload)
        )


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

    def __init__(self, model_id="raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L", retroactive_relations=False, irreflexive=True, quantize=False, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if quantize:
            bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, device_map="auto", quantization_config=bnb_config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, device_map="auto")
        self.retroactive_relations = retroactive_relations
        self.irreflexive = irreflexive
        self.arguments = dict()

    @torch.no_grad()
    def _classify_pairs(self, text_pairs):
        inputs = self.tokenizer(
            text_pairs,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        labels = [self.model.config.id2label[p.item()] for p in preds]
        return labels

    def _check_committed(self, iu, committed):
        return (iu.source in committed or self.arguments[iu.source.created_at][1]) and (iu.target in committed or self.arguments[iu.target.created_at][1])

    def process_update(self, update_message):
        um = retico_core.UpdateMessage()
        revoked = []
        added = []
        committed = []

        # Manage current argument status
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.REVOKE:
                revoked.append(iu)
                del self.arguments[iu.created_at]
            elif ut == retico_core.UpdateType.ADD:
                added.append(iu)
                self.arguments[iu.created_at] = (iu, False)
            else:
                if iu.created_at in self.arguments:
                    committed.append(iu)
                else:
                    added.append(iu)
                self.arguments[iu.created_at] = (iu, True)

        # Handle existing ius
        remove_ius = []
        for iu in self.current_output:
            if iu.get_source() in revoked or iu.get_target() in revoked:
                remove_ius.append(iu)
                um.add_iu(iu, retico_core.UpdateType.REVOKE)
            elif self._check_committed(iu, committed):
                remove_ius.append(iu)
                um.add_iu(iu, retico_core.UpdateType.COMMIT)
        self.current_output = [
            iu for iu in self.current_output if iu not in remove_ius]

        # Handle new ius
        for iu in added:
            new_argument = self.arguments[iu.created_at]
            child_to_parent_tuples = [(argument, new_argument) for argument in self.arguments.values(
            ) if not self.irreflexive or new_argument != argument]
            # TODO retro
            # parent_to_child_tuples = None if not self.retroactive_relations else [(new_argument, argument) for argument in self.arguments.values() if not self.irreflexive or new_argument != argument]
            if child_to_parent_tuples:
                child_to_parent = self._classify_pairs(
                    [(a[0][0].get_text(), a[1][0].get_text()) for a in child_to_parent_tuples])
                for i, label in enumerate(child_to_parent):
                    if label == "Conflict" or label == "Inference":
                        new_argument = child_to_parent_tuples[i][1]
                        argument = child_to_parent_tuples[i][0]
                        # If both source and target are committed, then we can safely commit
                        ut = retico_core.UpdateType.COMMIT if new_argument[
                            1] and argument[1] else retico_core.UpdateType.ADD
                        output_iu = self.create_iu(new_argument[0])
                        output_iu.set_relation(label)
                        output_iu.set_source(new_argument[0])
                        output_iu.set_target(argument[0])
                        if ut == retico_core.UpdateType.ADD:
                            self.current_output.append(output_iu)
                        um.add_iu(output_iu, ut)

        if len(um) > 0:
            self.append(um)
