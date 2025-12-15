from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

# Load args from yaml and merge with command line arguments
base_args = OmegaConf.load('ctc_args.yaml')
cli_args = OmegaConf.from_cli()
args = OmegaConf.merge(base_args, cli_args)

trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()