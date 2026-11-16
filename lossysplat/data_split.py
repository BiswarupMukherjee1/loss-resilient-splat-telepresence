"""Which sequences are used for what. Sequences are never split by frame.

The twelve UVG-VPC sequences show two people. One appears in four outfits (casual, flower dress, black
dress, sportswear); the other appears in the three "Blue" sequences and in ReadyForWinter. Holding out
the second person tests an unseen person, unseen clothes and unseen motion at once; holding out the
casual outfit tests unseen clothes on a seen person.
"""
TRAIN = ["ElegantDance", "FlowerDance", "FlowerWave", "Gymnast", "HelloGoodbye"]
VAL = ["ElegantWave"]                                   # seen person and outfit, unseen motion
TEST_UNSEEN_OUTFIT = ["CasualSpin", "CasualSquat"]      # seen person, unseen outfit
TEST_UNSEEN_PERSON = ["BlueBackpack", "BlueSpin", "BlueSquat", "ReadyForWinter"]
TEST = TEST_UNSEEN_OUTFIT + TEST_UNSEEN_PERSON
ALL = TRAIN + VAL + TEST
