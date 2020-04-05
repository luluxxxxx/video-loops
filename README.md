# video-loops

**automatic looping of image sequence using optical flow**

 - C++/OpenCV (> 4.2.0)
 - tested on Ubuntu 18.04 (KDE Neon)
 
![method](./readme_files/ScottWalker.gif)

**methodology :**

choose first and last frame of the loop (loop zone).
(to get invisible results they obviously must be as similar as possible).
cut in half and swap the 2 parts.
(the offensive cut is now in the middle of the loop).
fade A towards B around the middle frame.
(notice that you will use some frames before and after the loop)

![method](./readme_files/loops_method1s.jpg)

**using opticalflow for the fade :** 

bidirectional opticalflow vector fields from pair frames of A/B are generated in .flo format.they are used to warp frameA to frameB and vice-versa.each warped frames are then blended , generating a flow driven morphing.

![method](./readme_files/loops_method2s.jpg)

**Opticalflow methods :**

*OpenCV : implemented most of opencv proposed methods*
 -    ocv_deepflow
 -    farneback
 -    tvl1
 -    simpleflow
 -    sparsetodense
 -    rlof_epic
 -    rlof_ric
 -    pcaflow
 -    DISflow
 
*deepflow/deepmatch*

i'm usually getting the best morphing using deepflow/deepmatch method. I included the two static builds in this repository.they should work out of the box on Linux.(paths to executables are hardcoded in the code source , you should adjust it before compiling)
 
**installation**

