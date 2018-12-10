# Smiler

Smile detection with Keras 😄. Uses OpenCV to detect and crop faces and Keras for trained model to detect if 
the face is smiling.

### Smiling

<img src="https://github.com/sarvasvkulpati/Smiler/blob/master/images/smiling.jpg" width="500" alt="smilin' weird lookin' dude">

```
Neutral ------------------------------------###-------------- Smiling

```

### Neutral

<img src="https://github.com/sarvasvkulpati/Smiler/blob/master/images/neutral.jpg" width="500" alt="weird lookin' dude">

```
Neutral ###-------------------------------------------------- Smiling

```



## Getting Started

1. Download the project on your computer.
`
git clone https://github.com/sarvasvkulpati/Smiler
`
or download the ZIP file

2. Go to the directory with the file: ``` cd Smiler ```

3. Download the required packages: ``` pip install -r requirements.txt ```

4. Run the project: ``` python smile.py ```

## Raspberry Pi Version

I've also set up a version of this script that works with the Raspberry Pi, to use it, just replace smile.py 
with smileRaspi.py in the getting started instructions above. You can run it with a webcam and some LED's so that it 
detects how long you've been smiling!
