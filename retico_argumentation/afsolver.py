import json
import os
import retico_core

from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from py_arg.abstract_argumentation_classes.argument import Argument
from py_arg.abstract_argumentation_classes.defeat import Defeat
from py_arg.algorithms.semantics.get_grounded_extension import get_grounded_extension
from py_arg.algorithms.semantics.get_preferred_extensions import get_preferred_extensions
from py_arg.import_export.writer import Writer
from retico_argumentation.rstparser import RSTIU
from retico_argumentation.rbam import ArgumentRelationIU

# Making a custom writer that doesn't use pyargs awful fixed data output directory


class AFWriter(Writer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_dict(argumentation_framework: AbstractArgumentationFramework):
        return {'name': argumentation_framework.name,
                'arguments': [str(argument) for argument in argumentation_framework.arguments],
                'defeats': [(str(defeat.from_argument), str(defeat.to_argument))
                            for defeat in argumentation_framework.defeats]}

    def write(self, argumentation_framework: AbstractArgumentationFramework, filename: str):
        result = self.to_dict(argumentation_framework)
        with open(filename, 'w') as write_file:
            json.dump(result, write_file)


class ArgumentDetails:
    def __init__(self, text, speaker, tree=None):
        self.text = text
        self.speaker = speaker
        self.tree = tree
        self.relations = {"Rephrase": set(), "Conflict": set()}

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
        for label in self.relations:
            self.relations[label].discard(target)

    def substitute_relation(self, old_target, new_targets, label):
        if old_target in self.relations[label]:
            self.relations[label].discard(old_target)
            for new_target in new_targets:
                self.add_relation(new_target, label)

    def substitute_relations(self, old_target, new_targets):
        for label in self.relations:
            self.substitute_relation(old_target, new_targets, label)

    def __repr__(self):
        return f"Speaker {self.speaker}, Text: {self.text}, Relations:\n\tRephrase: {self.relations['Rephrase']}\n\tConflict: {self.relations['Conflict']}"


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
        output_dir='results',
        # Preferred "PR", or grounded "GR" solutions
        semantics="PR",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.semantics = semantics
        # Arguments by utterance timestamp, containing text, speaker, rst tree, and outgoing relations
        self.arguments = dict()
        self.banned = []

    def _rephrase_arguments(self):
        # If this is a rephrase of another argument, copy all relations over, delete this argument and any occuring relations
        remove_arguments = []
        for key, value in self.arguments.items():
            rephrases = value.get_relations("Rephrase")
            if len(rephrases) > 0:
                remove_arguments.append(key)
                for argument in self.arguments.values():
                    argument.substitute_relations(key, rephrases)
        for key in remove_arguments:
            del self.arguments[key]

    def _rename_arguments(self):
        # Create pretty keys
        speaker_counts = dict()
        new_arguments = dict()
        for key, value in self.arguments.items():
            speaker = value.get_speaker()
            if speaker in speaker_counts:
                speaker_counts[speaker] = speaker_counts[speaker] + 1
            else:
                speaker_counts[speaker] = 1
            new_name = f"{speaker}{speaker_counts[speaker]}"
            for argument in self.arguments.values():
                argument.substitute_relations(key, [new_name])
            new_arguments[new_name] = value
        self.arguments = new_arguments

    def _create_af(self):
        # Create arguments
        arguments = {name: Argument(name) for name in self.arguments}
        # Create attacks
        attacks = []
        for source, argument in self.arguments.items():
            for attacked in argument.get_relations("Conflict"):
                attacks.append(Defeat(arguments[source], arguments[attacked]))
        # Find solutions
        af = AbstractArgumentationFramework('af', arguments.values(), attacks)
        exts = []
        if self.semantics == "GR":
            exts = get_grounded_extension(af)
        elif self.semantics == "PR":
            exts = get_preferred_extensions(af)
        return af, arguments, exts

    def _output_rst_trees(self, argument_objs, solutions):
        for name, argument in self.arguments.items():
            filename = name
            argument_obj = argument_objs[name]
            for i, solution in enumerate(solutions):
                if argument_obj in solution:
                    filename = filename + f" ({self.semantics}{i})"
            argument.output_tree(os.path.join(
                self.output_dir, filename + '.rs3'))

    def process_update(self, update_message):
        for iu, ut in update_message:
            if isinstance(iu, RSTIU):
                origin = iu.grounded_in.created_at
                if ut == retico_core.UpdateType.REVOKE:
                    # Remove argument and any relations it occurs in
                    del self.arguments[origin]
                    for argument in self.arguments.values():
                        argument.delete_relations(origin)
                    self.banned.append(origin)
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
                elif source.created_at not in self.banned and target.created_at not in self.banned:
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
        print('--------------------')

    def shutdown(self):
        self._rephrase_arguments()
        self._rename_arguments()
        af, argument_objs, solutions = self._create_af()
        self._output_rst_trees(argument_objs, solutions)
        writer = AFWriter()
        writer.write(af, os.path.join(self.output_dir, 'af.json'))
