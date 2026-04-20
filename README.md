# slayminton
Badminton game analysis project that uses computer vision to track shuttlecocks and players.

# Workflow
`main.py`: The main loop for all functionalities.
`video_io.py`: Video input (compressed/resized to 720p). We use cv.MOG2 to identify the moving pixels and mask all non-moving pixels. Outputs an array of RGB image tensor. 
`dino.py`: tracking shuttle and players. Also contains the training loop for DINOv3 for shuttle tracking. Outputs an array of timestamps and coordinates that is 
`game_state.py`: determines the game state at each timestamp. Classification of shots based on output of `dino.py`. Outputs an array of timestamps, game state, who hit the shot, what shot was hit.
`analysis.py`: higher-level analysis of the game based on `game_state.py` and `shot_class.py`. Stores useful data as CSV files.
`visualization.py`: provides visualization of the data. Collaborates with `video_io.py` to create annotations on the original video as final output.

