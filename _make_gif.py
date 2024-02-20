import torchvision
import os
import imageio
import glob
imagefiles = glob.glob("./out/*.png")
imagefiles.sort()
imagefiles.reverse()

images = []

# group by digit
while len(imagefiles) > 0:
    images_at_i = [imagefiles.pop() for _ in range(10)]
    images_at_i = [torchvision.io.read_image(im) for im in images_at_i]
    image = torchvision.utils.make_grid(images_at_i,nrow=5)

    images.append(image)

total_duration_ms = 20000
duration_per_frame_ms = total_duration_ms / len(images)
w = imageio.get_writer('out.mp4',
                       #codec='hevc_vaapi',
                       format='FFMPEG', mode='I',
                       codec='h264',
                       fps = 1 / (duration_per_frame_ms / 1000))
for im in images:
    w.append_data(im.movedim(0, -1).numpy())
