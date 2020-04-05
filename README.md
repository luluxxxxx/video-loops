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
 
I'm usually getting decent results with ocv_deepflow.(made it the default method).but sometimes it's worth trying with other methods.you can have good surprises...

*deepflow/deepmatch*

i'm usually getting the best morphing using deepflow/deepmatch method. I included the two static builds in this repository.they should work out of the box on Linux.(paths to executables are hardcoded in the code source , you should adjust it before compiling).you may need lipng12.so.0 to run the deep statics on Ubuntu 18.04+.

beware that you should downscale your source images before using deepflow/deepmatch. I usually divide input resolution by 2 or 3.(see [flowscale] parameter) or it could take a loooong time to process and even crash.
 
**installation/compilation**

```sh
cd /toyourinstallationpath
git clone https://github.com/luluxxxxx/video-loops.git
cd video-loops
mkdir build
cd build
cmake ..
make
```
