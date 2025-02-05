import os
import ffmpeg
import subprocess

framerate = 5


out_dir = os.path.join("/cluster/scratch/daizhang/Wheat-GS-output-scaled/20240717/plot_461", "wheat-head", "run_test")
render_path = os.path.join(out_dir, f"01_a_360")
output_video = os.path.join(os.path.dirname(render_path), "test.mp4")

# subprocess.run("module load stack/2024-04 && module load ffmpeg/4.4.1", shell=True)
# subprocess.run([
#     "ffmpeg",
#     "-loglevel", "error",
#     "-framerate", str(framerate),
#     "-start_number", "0",
#     "-i", "%05d.png",  
#     "-vf", "scale=iw-mod(iw\,2):ih-mod(ih\,2)",
#     "-r", str(framerate),
#     "-vcodec", "libx264",
#     "-pix_fmt", "yuv420p",
#     output_video
#     ], cwd=render_path)

try:
    (
        ffmpeg
        .input(f"{render_path}/%05d.png", framerate=framerate, start_number=0)
        .filter("scale", "iw-mod(iw,2)", "ih-mod(ih,2)")  # Scale filter
        .output(output_video, r=framerate, vcodec="libx264", pix_fmt="yuv420p")
        .global_args("-loglevel", "error")  # Set global log level
        .run(capture_stdout=True, capture_stderr=True)
    )
    print("Video created successfully!")
except ffmpeg.Error as e:
    print("Error during FFmpeg execution:")
    print("STDERR:", e.stderr.decode())