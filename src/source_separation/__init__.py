from .unet.tampura import separate_tampura

__all__ = ["separate_tampura"]

# ftanet_predict requires the compiam conda environment — run explicitly:
# /home/lluis/anaconda3/envs/compiam/bin/python -m src.pitch_extraction.ftanet_predict <id>
