# video-loops

**automatic looping of image sequence using optical flow**

 - C++/OpenCV (> 4.2.0)
 - tested on Ubuntu 18.04 (KDE Neon)
 
![method](./readme_files/ScottWalker.gif)

**methodology :**

choose first and last frame of the loop (loop zone).
(to get invisible results they obviously must be as similar as possible.)
cut in half and swap the 2 parts.
(the offensive cut is now in the middle of the loop)
fade A towards B around the middle frame.
(notice that you will use some frames before and after the loop)

![method](./readme_files/loops_method1s.jpg)

**using opticalflow for the fade :** 

![method](./readme_files/loops_method2s.jpg)

**Opticalflow methods :**

 - opencv
 - deepflow/deepmatch
 
**installation**

