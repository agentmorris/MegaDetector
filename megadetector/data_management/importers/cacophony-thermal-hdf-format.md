# New Zealand Wildlife Thermal Imaging HDF format

## Clips

Each clip is stored in its own hdf file, named acorrding to [clip-id].hdf5 .  All clips have a frame rate of 9 frames per second.

Each clip has the following attributes:

* clip_id - unique identifier of this clip
* station_id - unique identifier of this clips location
* crop_rectangle - rectangle used to crop the clip (sometimes the edge of frames can get bad pixel values so we crop a few columns and rows), as [left, top, right, bottom].

The following are calculated on the cropped frames using the crop rectangle:

* frame_temp_max - hottest pixel per frame
* frame_temp_mean - median per frame
* frame_temp_min - coldest pixel per frame
* frames_per_second - fps of clip (always 9.0 in this dataset)
* max_temp - hottest pixel over clip
* mean_temp - mean pixel temperature over clip
* min_temp - coldest pixel over clip
* start_time - date/time the video started 
* temp_thresh - temperature threshold used when motion was detected
* ffc_frames - frame indices where calibration was running
* res_x - width in pixels
* res_y - height in pixels
* model - camera model name, "lepton3.5" or "lepton3"
* tags - array of strings describing any noteworthy properties of the recording, e.g. bad tracking, missing track, multiple animals, or interesting
* num_frames - number of thermal frames
* frames - dataset of frame data
* background - estimated thermal background frame (not always present)
* thermals - array of raw thermal frames
* tracks - array of track data (see below)

## Tracks

Each track has the following properties:

* id - unique track identifier for this clip
* ai_tag - the tag suggested by the AI model (all tags in this dataset have been added or verified by humans)
* ai_tag_confidence - AI confidence [0-1] 
* human_tag - the best human tag (if conflicting tags exist this will be None)
* human_tag_confidence - the human tag confidence
* human_tags - array of (label, confidence) of all human tags
* start_frame - frame number this track started
* end_frame - frame number this track ended
* positions - array of position information as a list of [left,top,right,bottom,frame_number,mass, blank_frame] tuples. "blank_frame" is either 0 or 1, indicating whether the region was predicted from previous regions during this frame.  "mass" is the number of non zero pixels in the tracked object above a threshold after a simple blur.<br/><br/>For example:<br/><br/>[0, 10, 10, 20, 1, 16, 0] - 10 x 20 rectangle at frame 1 with mass 16<br/>
 [0, 10, 10, 20, 2, 0, 1] - A predicted region using previous momentum of 10 x 20 rectangle at frame 2 with mass 0

