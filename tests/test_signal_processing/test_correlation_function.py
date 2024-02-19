
from waveanalysis.signal_processing.correlation_functions import acf_shifts

def test_acf_shifts_some_condition():
    known_thing = 'known thing'
    exp_thing = acf_shifts(data)
    assert exp_thing == known_thing

def test_acf_shifts_some_other_condition():
    known_thing = 'known thing'
    exp_thing = acf_shifts(data)
    assert exp_thing == known_thing
