import datetime
import cv2
import os
def compile_video():
    """Compiles saved images into a video."""
    frame_size = (640, 480)  # Size of the video frame
    fps = 60  # Frames per second
    video_filename = "tensorboard_video_{}.mp4".format(
        datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H-%M-%S"))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    # Write images to video
    for epoch in range(2300):  # Adjust range according to your epochs
        img_path = f'../images/epoch_{epoch}.png'
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, frame_size)  # Resize to fit video frame
            out.write(img)  # Write the frame

    out.release()
    cv2.destroyAllWindows()
    print(f'Video saved as: {video_filename}')



if __name__  == "__main__":
    compile_video()