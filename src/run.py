from detection.config import Config
from detection.pipeline import Pipeline

if __name__ == "__main__":
    cfg      = Config()
    pipeline = Pipeline(cfg)
    pipeline.run()