Low Cost Computational Astrophotography
Joseph Yen and Peter Bryan

There are three matlab files in this folder: astrophotography.m, compressStars.m, and recAdd.m . astrophotography.m is the main script, and takes a user-specified image with star streaks and transforms it into a corresponding image with properly localized point stars. This is this program you should run. compressStars is a helper function that, when passed an image consisting only of star streaks, compresses the streaks into point stars. recAdd is a helper function for compressStars, and recursively gathers the bright pixels in a streak into a cluster.

Tl;dr: run astrophotography.m!
