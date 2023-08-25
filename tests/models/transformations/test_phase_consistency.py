"""
Tests for the phase consistency transformations
"""
import numpy.random

from cdl_eeg.models.transformations.phase_consistency import BivariateTimeShift, BivariatePhaseShift
from cdl_eeg.models.transformations.utils import chunk_eeg, UnivariateNormal


# ------------------------------
# Bivariate Time Consistency
# ------------------------------
def test_bivariate_time_consistency():
    """Test of transformation. Testing type, output shapes, that the correct chunks are permuted, and that no in-place
    operations altered the input"""
    # -------------------
    # Hyperparameters
    # -------------------
    batch_size, num_time_steps = 10, 5000
    std, distribution = 50, "normal"
    num_chunks, chunk_duration, chunk_time_delay = 4, 1000, 200

    # -------------------
    # Generate data
    # -------------------
    # Two univariate signals
    x0 = numpy.random.normal(size=(batch_size, num_time_steps))
    x1 = numpy.random.normal(size=(batch_size, num_time_steps))

    # Make copies (used for testing if any in-place operations happen)
    x0_copy = x0.copy()
    x1_copy = x1.copy()

    # -------------------
    # Perform bivariate time shift transformation
    # -------------------
    transformation = BivariateTimeShift(std, distribution, num_chunks=num_chunks, chunk_duration=chunk_duration,
                                        chunk_time_delay=chunk_time_delay)

    data = numpy.concatenate((numpy.expand_dims(x0, axis=1), numpy.expand_dims(x1, axis=1)), axis=1).copy()
    original = chunk_eeg(data, k=num_chunks, chunk_duration=chunk_duration, delta_t=chunk_time_delay,
                         chunk_start_shift=0)
    permuted, idx, _ = transformation.phase_shift(x0, x1, permute_first_channel=True)

    # -------------------
    # Tests
    # -------------------
    # Type checking
    assert isinstance(permuted, tuple), (f"Expected the permuted chunks to be of type 'tuple', but found "
                                         f"{type(permuted)}")
    assert all(isinstance(chunk, numpy.ndarray) for chunk in permuted), \
        f"Expected the chunks to be of type 'numpy.ndarray', but found {tuple(set(type(chunk) for chunk in permuted))}"

    # Check if the permuted chunk and channel was actually permuted (first channel was permuted)
    assert not numpy.array_equal(permuted[idx][:, 0], original[idx][:, 0]), \
        "Expected the permuted chunk to be different from the original, but they were equal"

    # Check if non-permuted chunks and channel are the same
    for i, (permuted_chunk, original_chunk) in enumerate(zip(permuted, original)):
        # Special handling for the permuted chunk
        if i == idx:
            # The second channel should be the same
            assert numpy.array_equal(permuted_chunk[:, 1], original_chunk[:, 1]), \
                ("Expected the non-permuted channel in the permuted chunk to be the same as the original, but they "
                 "were different")
            continue

        # Check if the chunks are similar as expected
        assert numpy.array_equal(permuted_chunk, original_chunk), \
            f"Expected the non-permuted chunk to be the same as the original, but they were different (chunk nr. {idx})"

    # Check if the numpy arrays were changed in-place
    assert numpy.array_equal(x0, x0_copy), "Expected the signal to be unaltered, but in-place operations took place"
    assert numpy.array_equal(x1, x1_copy), "Expected the signal to be unaltered, but in-place operations took place"


# ------------------------------
# Bivariate Phase Consistency
# ------------------------------
def test_bivariate_phase_shift_transformation():
    """Test of transformation. Testing type, output shapes, that the correct chunks are permuted, and that no in-place
    operations altered the input"""
    # -------------------
    # Hyperparameters
    # -------------------
    batch_size, num_time_steps = 10, 5000
    phase_distribution = UnivariateNormal(0, 10*2*numpy.pi)
    num_chunks, chunk_duration, chunk_time_delay = 4, 1000, 200

    # -------------------
    # Generate data
    # -------------------
    # Two univariate signals
    x0 = numpy.random.normal(size=(batch_size, num_time_steps))
    x1 = numpy.random.normal(size=(batch_size, num_time_steps))

    # Make copies (used for testing if any in-place operations happen)
    x0_copy = x0.copy()
    x1_copy = x1.copy()

    # -------------------
    # Perform bivariate phase consistency
    # transformation
    # -------------------
    transformation = BivariatePhaseShift(phase_distribution, num_chunks=num_chunks, chunk_duration=chunk_duration,
                                         chunk_time_delay=chunk_time_delay)

    data = numpy.concatenate((numpy.expand_dims(x0, axis=1), numpy.expand_dims(x1, axis=1)), axis=1).copy()
    original = chunk_eeg(data, k=num_chunks, chunk_duration=chunk_duration, delta_t=chunk_time_delay,
                         chunk_start_shift=0)
    permuted, idx, _ = transformation.phase_shift(x0, x1, permute_first_channel=True)

    # -------------------
    # Tests
    # -------------------
    # Type checking
    assert isinstance(permuted, tuple), (f"Expected the permuted chunks to be of type 'tuple', but found "
                                         f"{type(permuted)}")
    assert all(isinstance(chunk, numpy.ndarray) for chunk in permuted), \
        f"Expected the chunks to be of type 'numpy.ndarray', but found {tuple(set(type(chunk) for chunk in permuted))}"

    # Check if the permuted chunk and channel was actually permuted (first channel was permuted)
    assert not numpy.array_equal(permuted[idx][:, 0], original[idx][:, 0]), \
        "Expected the permuted chunk to be different from the original, but they were equal"

    # Check if non-permuted chunks and channel are the same
    for i, (permuted_chunk, original_chunk) in enumerate(zip(permuted, original)):
        # Special handling for the permuted chunk
        if i == idx:
            # The second channel should be the same
            assert numpy.array_equal(permuted_chunk[:, 1], original_chunk[:, 1]), \
                ("Expected the non-permuted channel in the permuted chunk to be the same as the original, but they "
                 "were different")
            continue

        # Check if the chunks are similar as expected
        assert numpy.array_equal(permuted_chunk, original_chunk), \
            f"Expected the non-permuted chunk to be the same as the original, but they were different (chunk nr. {idx})"

    # Check if the numpy arrays were changed in-place
    assert numpy.array_equal(x0, x0_copy), "Expected the signal to be unaltered, but in-place operations took place"
    assert numpy.array_equal(x1, x1_copy), "Expected the signal to be unaltered, but in-place operations took place"
