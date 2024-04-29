import pytest


@pytest.fixture
def rbp_model_configs():
    """Configurations for building a MainRBPModel from config"""
    # RBP config
    rbp_config = {
        "RBPDesigns": {
            "RBPDesign0": {
                "cmmn_kwargs": {
                    "kernel_size": 128
                },
                "num_designs": 1,
                "pooling_methods": "MultiMSMean",
                "pooling_methods_kwargs": {},
                "pooling_type": "multi_cs",
                "split_methods": ["CentroidPolygons"],
                "split_methods_kwargs": [{
                    "channel_positions": ["Miltiadous"],
                    "k": [2, 2, 2, 2, 2, 2, 2, 2],
                    "min_nodes": 1
                }],
                "use_cmmn_layer": True
            },
            "RBPDesign1": {
                "cmmn_kwargs": {
                    "kernel_size": 512
                },
                "num_designs": 1,
                "pooling_methods": "MultiMSSharedRocketHeadRegion",
                "pooling_methods_kwargs": {
                    "bias": False,
                    "latent_search_features": 12,
                    "max_receptive_field": 123,
                    "num_kernels": 203,
                    "share_search_receiver_modules": [True, False]
                },
                "pooling_type": "multi_cs",
                "split_methods": ["CentroidPolygons"],
                "split_methods_kwargs": [{
                    "channel_positions": ["Miltiadous"],
                    "k": [2, 2, 2, 2, 2, 2, 2, 2],
                    "min_nodes": 2
                }],
                "use_cmmn_layer": True
            },
            "RBPDesign2": {
                "cmmn_kwargs": {
                    "kernel_size": 128
                },
                "num_designs": 1,
                "pooling_methods": "MultiMSMean",
                "pooling_methods_kwargs": {},
                "pooling_type": "multi_cs",
                "split_methods": ["CentroidPolygons"],
                "split_methods_kwargs": [{
                    "channel_positions": ["Miltiadous"],
                    "k": [2, 3, 2, 3, 2, 3, 2, 3, 2],
                    "min_nodes": 2
                }],
                "use_cmmn_layer": True
            }
        },
        "normalise_region_representations": True}

    # MTS module config
    mts_config = {"model": "InceptionNetwork", "kwargs": {"depth": 12, "cnn_units": 31, "num_classes": 1}}

    return rbp_config, mts_config
