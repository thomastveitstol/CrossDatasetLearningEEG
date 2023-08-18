from cdl_eeg.models.dummy_test import hello_world


def test_hello_world():
    assert hello_world() == "hello world"
