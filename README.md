# Vehicle-Tracker
Ever tried to track an object in a video, only for the tracker to lose it the second it drives behind a tree or another car? It’s a common frustration in computer vision.
This script fixes that. It combines a standard OpenCV visual tracker (CSRT) with a Kalman Filter to keep the bounding box locked on your target—even if the view gets blocked for a moment.
Why this is useful?
Most basic trackers rely entirely on what they "see." If the target gets covered up (occlusion), the tracker panics and fails.
My approach uses a "Physics Helper":
When the tracker can see the object clearly, it uses that data.
If the object disappears or the tracker gets confused, the script switches to a Kalman Filter. It uses math to calculate where the car should be based on its previous speed and direction.
Essentially, it remembers the car’s momentum, so the bounding box keeps moving smoothly even when the target is temporarily hidden.

1. What you need
You’ll need Python and a couple of OpenCV libraries.

2. How to run it
Place the video file you want to analyze (e.g., car_3.mp4) in the same folder as the script.
Open the code and update the video_path variable to match your filename.
Run the code in your choosen pkatform I have ran it on vscode.

3. Controls
*   Selecting the target: When the video starts, a window will pop up. Just draw a box around the vehicle you want to track and press **Enter**.
*   Stopping: Press the **'q'** key at any time to close the window and end the program. q belongs to quit.

4. How it works (The Simple Version)
*   Predict: Before looking at the frame, the Kalman Filter "guesses" where the car will be based on its past speed.
*   Observe: The CSRT tracker tries to find the car in the new frame.
*   Verify: The code compares the "Guess" vs. the "Observation." If they are close, it trusts the tracker. If they are far apart (a "jump"), it assumes the tracker is lost and switches to the Kalman Filter's prediction until the target reappears
