# baltic-scripts
This repository contains the scripts for the SeaLaBio project in the frame of Baltic+

## Execution
To start the processing use the RunFile.py
Adapt the paths within the file to your needs.

In particular, you need to adapt the variables *inpath* and *outpath*.
Also the *fnames* array needs to be adapted with the used files names.

If processing Sentinel-2, the current call to baltic_AC needs to be deactivated and the currently deactivated needs to be activated.
The *atmosphericAuxDataPath* variable needs to be updated in this case too.
