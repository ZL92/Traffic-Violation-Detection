# Ultra-Fast-Lane-Detection
Copy right: This is based on https://github.com/cfzd/Ultra-Fast-Lane-Detection. 
For details please check it

# Dependencies
pip install -r requirement

Choose your way to install torch. For example: 
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 

# Download pretrained model
Download the model from the following link and place it in the main folder

https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view?usp=sharing

# Batch running
Change main path in each python file to the path of videos, then run final_batch_processing.sh. 
The bash file works as the following:

1. mp4tolist.py # convert video into images and list txt file
2. batch_demo.py # generate ML result
3. whole_processing.py # generate vehicle masks
4. lane_fit.py # calculate line regression result based on ML result
5. only_mask_obj.py # use mask of objection and hough line method to get a pure CV result
6. final_lines_generator.py # generate final lane detection results based on the previous ML and CV results
7. solid_dot_lane.py # assign line_style label to the final results
8. line_crossing_detection.py # line crossing detection and output final json