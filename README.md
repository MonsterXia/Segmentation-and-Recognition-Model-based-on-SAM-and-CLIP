# Segmentation and Recognition Model based on SAM and CLIP



## Installation

Package version:

- Python = 3.9.16
- PyTorch = 2.0.0+cu118
- torchvision = 0.15.1+cu118
- Transformers = 4.29.2

Click the links below to download the part of model .

- [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)



## Notice

### Address

- The address of the pictures that need to be separated and recognized in the test: ./notebooks/images
- Main executable program location: ./src
- After downloading the ViT-H SAM model, please create a "model" folder in the root directory and place the model in it.
- If you need to validate on the test set, please create a "dataset" folder in the root directory and place the actual dataset folder under the "dataset" folder.



## Simple test and analysis

We have conducted preliminary analysis on the model, and you can find the corresponding results in the "analysis" folder. 



## License

The model is licensed under the MIT license.



## Reference

[1] A. Kirillov et al., "Segment anything," arXiv preprint arXiv:2304.02643, 2023.

[2] A. Radford et al., "Learning transferable visual models from natural language supervision," in International conference on machine learning, 2021: PMLR, pp. 8748-8763. 