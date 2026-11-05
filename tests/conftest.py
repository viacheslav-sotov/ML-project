def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: uses real LIME explanations and saved model files; exclude with -m \"not slow\"",
    )
