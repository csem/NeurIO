#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 17.10.23
Description: TODO
"""

from neurio.tasks.classification.keyword_spotting import SpeechCommands
#import tensorflow_datasets as tfds

# create a testsuite
def test_speech_commands():
    sc_dataset = SpeechCommands()
    train = sc_dataset.get_train_data()
    test = sc_dataset.get_test_data()
    val = sc_dataset.get_validation_data()

    print(len(train[0]))
    # assert datasets size correspond to SpeechCommands v0.02 dataset
    assert len(train[0]) == 51088
    assert len(test[0]) == 6798
    assert len(val[0]) == 6798



train_dataset = tfds.load("speech_commands", split="train", shuffle_files=True)
print(len(train_dataset))