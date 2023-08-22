import numpy
import pytest

from cdl_eeg.models.transformations.utils import chunk_eeg


# -----------------------
# Tests for chunking EEG
# -----------------------
def test_chuck_eeg():
    """Test if the chunk_eeg function works properly (returns correct data in a simple case)"""
    # ------------------------
    # Generate data todo: config file
    # ------------------------
    data = tuple(numpy.ones(shape=(10, 25, 300)) * (j + 1) for j in range(4))
    intervals = tuple(numpy.zeros(shape=(10, 25, 200)) for _ in range(3))
    unused_data = numpy.random.normal(loc=0, scale=5, size=(10, 25, 500))

    # Stack them
    chunked_data = [data[0]]
    for interval, chunk in zip(intervals, data[1:]):
        chunked_data.append(interval)
        chunked_data.append(chunk)

    chunked_data = numpy.concatenate(chunked_data, axis=-1)

    # Add unused data from both sides (in temporal dimension)
    chunked_data = numpy.concatenate((unused_data, chunked_data, unused_data), axis=-1)

    # ------------------------
    # Compare actual with expected
    # ------------------------
    actual_chunks = chunk_eeg(data=chunked_data, k=4, chunk_duration=300, delta_t=200)

    assert len(actual_chunks) == len(data), (f"The number of chunks was wrong. Expected {len(data)}, got "
                                             f"{len(actual_chunks)}")
    assert all(numpy.array_equal(actual, expected) for actual, expected in zip(actual_chunks, data)), \
        "The arrays were not as expected"


def test_chunk_eeg_error():
    """Test if an error is properly raised when the expected number of time steps for chunking exceeds the number of
    time steps available"""
    # Generate dummy data
    data = numpy.random.normal(size=(10, 25, 1250))

    # Check if the error is correctly raised
    expected_msg = ("The specified hyperparameters require EEG data with 1300 number of time steps, but only 1250 are "
                    "available.")
    with pytest.raises(AssertionError, match=expected_msg):
        chunk_eeg(data, k=4, chunk_duration=250, delta_t=100)
