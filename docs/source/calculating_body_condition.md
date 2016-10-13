# Estimating body density of cetaceans: the manual


Tomoko Narazaki (naratomoz@gmail.com)
Lucia Martina Martin Lopez (luciamartinaml@gmail.com)
12 June 2016


This document describes the sequence of operations needed to estimate body
density of whales from tag data using a hydrodynamic glide model. For this
analysis, it is needed either IGOR Pro (WaveMetrics,
https://www.wavemetrics.com) or Matlab (MathWorks,
http://www.mathworks.com/products/matlab/) for tag data processing. Then, R
Studio (https://www.rstudio.com) will be used for Bayesian estimation.


Data required for the analysis:

1. High-resolution tag data (Dtag, Little Leonardo 3MPD3GT etc.) 
   * 3-axis acceleration **essential** 
   * depth **essential**
   * speed *preferable*
   * 3-axis magnetism *preferable*
   * temperature *preferable*

2. CTD profile at the field site

## Quick outline

### Part 1
1. Calibrate > Whale frame > calc. pitch, roll, heading (prh in dtag)
2. Define dives: dives deeper then `y` meters
3. Separate gravity-based and specific acceleration
4. Separate dive into descent, bottom, ascent phases
5. Estimate swim speed: vert_speed/sin(pitch>30deg)
   * speed calculated in calibration step if tag has propeller
6. Estimate seawater density around tagged animal
7. Extract strokes and glides: high-pass filter -or- rotations with magnetometer
8. Make 5s sub-glides w/: mean depth, pitch, speed, acceleration etc.
9. Calculate glide ratio during descent and ascent phases: for general review


## Part 1: Tag data processing using either IGOR Pro or Matlab

1. Before the analysis, the tag data must be prepared as follows: Complete all
   calibrations (i.e. temperature drift of sensors, convert the recordings into
   proper units etc) Convert into whale-frame Calculate pitch, roll and heading
   (i.e. prh format in Dtag data)


IGOR Pro Users:
Save “Body density” folder that include all IGOR Pro macro required for body
density analysis into “User Procedure” folder in the IGOR Pro Folder.  To load
a txt file, open IGOR Pro, go to: ‘Data’  ‘Load waves’  ‘Load general text
file’.  Change interval of data to be based on sampling frequency by going to:
‘Data’  ‘Change wave scaling’; alter the delta value (e.g. for 5Hz, Δ=0.2 s)

In this analysis (using IGOR Pro), the waves are named as follows:

Wave names                Description
surgeAw, swayAw, heaveAw  Surge, sway & heave accelerations, respectively, at whale-frame
surgeMw, swayMw, heaveMw  Surge, sway & heave magnetism, respectively, at whale-frame
D                         Depth
pitch                     Pitch in degrees
roll                      Roll in degrees
head                      Heading in degrees

Matlab Users:
Save “Body density” folder and add it into your Matlab path. 
Use ‘File’  ‘Set path’ in the pull-down menu at the top of the Matlab screen and add the “Body density analysis” folder into your path. Press save button. 
Open “body_density” script and follow the instructions there.
Load prh. 

Wave names Dtagdata  Description
Aw                   3-axis accelerometer data at the whale frame; i.e.,
                     Aw(:,1) longitudinal x axis; (:,2) lateral y axis;
                     (:,3) dorso-ventral z axis.
Mw                   3-axis magnetometer data at the whale frame; i.e.,
                     Mw(:,1) longitudinal x axis; (:,2) lateral y axis;
                     (:,3) dorso-ventral z axis.
p                    Depth in meters
pitch                Pitch in radians
roll                 Roll in radians
head                 Heading in radians

Type help and the name of the function for a description of input and output of
each function; e.g. help finddives


2. Define dives – then, make a summary table describing the characteristics of
   each dive.  Here, dives are defined as any submergence deeper than 2 m (i.e.
   dive definition = 2 m). But, in this analysis, only deep dives of which max
   depth is bigger than 10m (i.e. deep dive definition = 10 m) will be extracted.
   Change the definitions appropriately according to your data.

IGOR Pro:
* Type #include “FindDives” in the procedure window.
* In the macro FindDives, set DiveDef (dive definition), D_DiveDef (Deep dive
  definition) and fs (sampling frequency of depth data). 
* Run the macro by typing FindDives (D) into the command window. 
* Type edit Dive in the command window to see the summary table
* After running the macro, the following waves will be created:


Wave names  Description
Dive[][0]   Start time of each dive (i.e. No of seconds since 1904/1/1)
Dive[][1]   End time of each dive (i.e. No of seconds since 1904/1/1)
Dive[][2]   Dive duration in seconds
Dive[][3]   Post-dive surface duration in seconds
Dive[][4]   Time of the deepest point of each dive (i.e. No of seconds since 1904/1/1) 
Dive[][5]   Max dive depth (m)
Start       Dive start point
Fin         Dive end point
DiveNumber  Dive ID. Starting from 1.
MaxD        Maximum dive depth (m)
Duration    Dive duration (s)
