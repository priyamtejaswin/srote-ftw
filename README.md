# Super Resolution, On The Edge, For The Win!
*by Akshay Chawla and Priyam Tejaswin*
```
.
├── README.md
├── papers ## PDFs for review and reference (not committed).
├── src_tensorflow ## Code in tf (using `tensorflow.keras`). 
└── src_tf2 ## Code in tf2_alpha.
```

## Dependencies
- `src_tensorflow` uses **tensorflow 1.13.1**. This is the latest stable release as of creating this repository.
- `src_tf2` uses **tensorflow 2.0** which is still an alpha build.

## System Design
![](system_blocks.jpg)

## References
- VESPCN implementation (unofficial, TF) - <https://github.com/LoSealL/VideoSuperResolution/blob/master/VSR/Models/Vespcn.py>
- Fully-featured TF2 implementation of YOLOv3 (for checking TF2 oddities) - <https://github.com/zzh8829/yolov3-tf2>