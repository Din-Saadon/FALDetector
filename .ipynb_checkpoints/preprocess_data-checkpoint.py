import os


#pred SpyNet
os.system("python modules/SpyNet/pred.py --model sintel-final --one ../sources/images/modified --two ../sources/images/reference --out ../sources/flows/fwd")
os.system("python modules/SpyNet/pred.py --model sintel-final --one ../sources/images/reference --two ../sources/images/modified --out ../sources/flows/bwd ")



