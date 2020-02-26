from pytracking.admin.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.dataset_path = ''  # Where benchmark datasets are stored
    settings.network_path = ''  # Where tracking networks are stored.
    settings.results_path = '/models/PaddleCV/tracking/pytracking/tracking_results/'  # Where to store tracking results

    return settings
