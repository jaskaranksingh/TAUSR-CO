
from core.pipeline import run_pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    run_pipeline(config_path=args.config)
