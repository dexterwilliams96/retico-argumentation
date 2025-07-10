import retico_core


from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from py_arg.abstract_argumentation_classes.argument import Argument
from py_arg.abstract_argumentation_classes.defeat import Defeat
from retico_argumentation.rstparser import RSTIU
from retico_argumentation.rbam import ArgumentRelationIU


class ArgumentDetails:
    def __init__(self, text, speaker, tree=None):
        self.text = text
        self.speaker = speaker
        self.tree = tree
        self.relations = {"Rephrase": set(), "Inference": set(),
                          "Conflict": set()}

    def get_text(self):
        return self.text

    def output_tree(self, filename):
        if self.tree is not None:
            self.tree[0].to_rs3(filename)
        else:
            print('Missing RST Tree!')

    def set_tree(self, tree):
        self.tree = tree

    def get_speaker(self):
        return self.speaker

    def set_speaker(self, speaker):
        self.speaker = speaker

    def get_relations(self, label):
        return self.relations[label]

    def add_relation(self, target, label):
        self.relations[label].add(target)

    def delete_relation(self, target, label):
        self.relations[label].discard(target)

    def delete_relations(self, target):
        self.relations["Rephrase"].discard(target)
        self.relations["Inference"].discard(target)
        self.relations["Conflict"].discard(target)

    def __repr__(self):
        return f"Speaker {self.speaker}, Text: {self.text}, Relations:\n\tRephrase: {self.relations['Rephrase']}\n\tInference: {self.relations['Inference']}\n\tConflict: {self.relations['Conflict']}"


class AFModule(retico_core.AbstractConsumingModule):
    """A module that consumes RSTTrees and argument relations to produce
    an abstract argument framework. On shutdown, it outputs the solutions and
    their associated RST trees"""

    @staticmethod
    def name():
        return "AF Module"

    @staticmethod
    def description():
        return "A consuming module that produces a Dung-style abstract argumentation framework."

    @staticmethod
    def input_ius():
        return [RSTIU, ArgumentRelationIU]

    @staticmethod
    def output_iu():
        return None

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Arguments by utterance timestamp, containing text, speaker, rst tree, and outgoing relations
        self.arguments = dict()

    def process_update(self, update_message):
        for iu, ut in update_message:
            if isinstance(iu, RSTIU):
                origin = iu.grounded_in.created_at
                if ut == retico_core.UpdateType.REVOKE:
                    # Remove argument and any relations it occurs in
                    del self.arguments[origin]
                    for argument in self.arguments.values():
                        argument.delete_relations(origin)
                else:
                    # Add argument if it doesn't exist, otherwise add tree
                    if origin in self.arguments:
                        self.arguments[origin].set_tree(iu.get_tree())
                    else:
                        self.arguments[origin] = ArgumentDetails(
                            iu.grounded_in.get_text(), iu.grounded_in.get_speaker(), iu.get_tree())
            else:
                source = iu.get_source()
                target = iu.get_target()
                relation = iu.get_relation()
                if ut == retico_core.UpdateType.REVOKE and source.created_at in self.arguments:
                    # Remove the relation
                    self.arguments[source.created_at].delete_relation(
                        target.created_at, relation)
                elif source.created_at:
                    # Add source and target if they don't exist
                    if source.created_at not in self.arguments:
                        self.arguments[source.created_at] = ArgumentDetails(
                            source.get_text(), source.get_speaker())
                    if target.created_at not in self.arguments:
                        self.arguments[target.created_at] = ArgumentDetails(
                            target.get_text(), target.get_speaker())
                    self.arguments[source.created_at].add_relation(
                        target.created_at, relation)
        print('--------------------')
        for key, value in self.arguments.items():
            print(f'TS: {key}')
            print(value)
            print()

    def shutdown(self):
        # Map any rephrases to the same argument (also naively check content)
        pass
        # Map inferences to attacked
