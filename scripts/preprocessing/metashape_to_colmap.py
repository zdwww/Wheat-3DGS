"""
Camera alignment in Agisoft Metashape and export to COLMAP (undistorted pinhole, centred principal point), in a
projected CRS (default Swiss LV95, EPSG:2056). This is how the reprocessed version of our dataset was made. Afterwards run
recenter_colmap.py, which moves the origin to the mean camera centre (as in the released data) and saves the offset.
Optionally also builds the MVS dense point cloud (--mvs) and exports it as E57 in the same CRS.

Needs Metashape Professional with a license and its Python module
(wheel from https://www.agisoft.com/downloads/installer/, tested with 2.2.1).

Two modes:
    # align the images of a plot, georeference with the markers, save the project and export
    python scripts/preprocessing/metashape_to_colmap.py --images <plot>/images_raw --markers <plot>/plot_461_coordinates.txt \
        --project <out>/agisoft.psx --out <out>/colmap [--mvs]
    # export only, from an existing aligned project (opened read-only, nothing is re-aligned; with --mvs it is
    # opened for writing and the dense cloud is saved into it, so use a copy)
    python scripts/preprocessing/metashape_to_colmap.py --project <out>/agisoft.psx --out <out>/colmap [--no_images]

Markers: one file per plot with label,x,y,z per line in --crs ('#' lines are skipped); labels must match Metashape's
detected coded targets ("target 1", ...). gnss_to_markers.py writes these files from a GNSS (e.g. Trimble) survey.
Coordinates in a local metric system (e.g. tape measurements) work too, with --crs local; see gnss_to_markers.py.

The export writes <out>/images/<name>.png (undistorted), <out>/sparse/0/{cameras,images,points3D}.txt, the same cameras
with their distortion model (no images; they are the raw input images) in <out>/distorted/sparse/0/, and
<out>/distorted/marker_projections.csv; with --mvs also <out>/pointcloud.e57. The marker pixels are in the raw
(distorted) input images, as Metashape stores them: use them with the raw images and distorted/sparse/0, not with
images/ and sparse/0. recenter_colmap.py --distorted moves both models by the same offset.

Why pinhole + a projected CRS: the 3DGS camera model here ignores cx, cy (it assumes the principal point at the image
centre), and Metashape's pinhole conversion centres it. Always export with a projected CRS (--crs): without one,
Metashape's COLMAP export uses its own local frame (rotated and shifted relative to the markers' grid), and the
cam_centroid saved by recenter_colmap.py would not map the model back to that CRS.
"""
import os
from argparse import ArgumentParser

import Metashape

# Metashape's "Local Coordinates (m)", for markers measured in a local metric system
LOCAL_CRS = 'LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'


def export_marker_projections(chunk, path):
    """Marker pixel coordinates in the raw (distorted) images, as Marker,Camera,X,Y,Pinned (Pinned = detected, not projected)"""
    count = 0
    with open(path, "w") as f:
        f.write("Marker,Camera,X,Y,Pinned\n")
        for marker in chunk.markers:
            for camera, proj in marker.projections.items():
                f.write(f"{marker.label},{camera.label},{proj.coord.x},{proj.coord.y},{proj.pinned}\n")
                count += 1
    return count


def align(images_dir, markers_path, crs, exclude):
    """Same steps as for our reprocessed dataset: detect the coded targets, georeference with their coordinates, align at full resolution"""
    doc = Metashape.Document()
    chunk = doc.addChunk()
    images = sorted(os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".tif")))
    images = [p for p in images if not any(os.path.splitext(p)[0].endswith(f"_{e}") for e in exclude)]
    chunk.addPhotos(images)
    print(f"Added {len(images)} images from {images_dir}")

    chunk.detectMarkers(tolerance=100)
    chunk.importReference(markers_path, format=Metashape.ReferenceFormatCSV, delimiter=",", columns="nxyz", crs=crs)
    chunk.crs = crs
    chunk.updateTransform()

    chunk.matchPhotos(downscale=0, reference_preselection_mode=Metashape.ReferencePreselectionSequential)
    chunk.alignCameras()
    return doc, chunk


def build_mvs(chunk, path, crs):
    """Dense point cloud as for the paper's MVS clouds (full-resolution depth maps, default filtering), saved as E57"""
    chunk.buildDepthMaps(downscale=1)
    chunk.buildPointCloud()
    chunk.exportPointCloud(path, format=Metashape.PointCloudFormatE57, crs=crs)
    print(f"Exported MVS point cloud to {path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Metashape alignment + COLMAP export in a projected CRS")
    parser.add_argument("--project", type=str, required=True, help="Metashape project (.psx): written with --images, else read")
    parser.add_argument("--out", type=str, required=True, help="output folder for images/, sparse/0/ and marker_projections.csv")
    parser.add_argument("--images", type=str, default=None, help="raw (demosaiced) images to align; omit to export from --project")
    parser.add_argument("--markers", type=str, default=None,
                        help="marker coordinates file (label,x,y,z in --crs), needed with --images; from a GNSS survey: gnss_to_markers.py")
    parser.add_argument("--crs", type=str, default="EPSG::2056",
                        help="CRS of the markers and the export: EPSG code, WKT, or 'local' for local metres (default: Swiss LV95)")
    parser.add_argument("--exclude", type=str, nargs="*", default=["cam_13"], help="image name suffixes to leave out (camera 13 for our plots)")
    parser.add_argument("--no_images", action="store_true", help="only write sparse/ (e.g. when the undistorted images exist already)")
    parser.add_argument("--mvs", action="store_true", help="also build the dense point cloud and export <out>/pointcloud.e57")
    parser.add_argument("--gpu_mask", type=int, default=None, help="Metashape GPU bitmask, e.g. 2 = second GPU only (default: Metashape's setting)")
    args = parser.parse_args()

    crs = Metashape.CoordinateSystem(LOCAL_CRS if args.crs == "local" else args.crs)
    if args.gpu_mask is not None:
        Metashape.app.gpu_mask = args.gpu_mask
    if args.images is not None:
        assert args.markers is not None, "--images needs --markers (from a GNSS survey: gnss_to_markers.py)"
        assert not os.path.exists(args.project), f"{args.project} exists; omit --images to export from it"
        doc, chunk = align(args.images, args.markers, crs, args.exclude)
        os.makedirs(os.path.dirname(os.path.abspath(args.project)), exist_ok=True)
        doc.save(args.project)
        print(f"Saved {args.project}")
    else:
        doc = Metashape.Document()
        doc.open(args.project, read_only=not args.mvs, ignore_lock=not args.mvs)
        chunk = doc.chunk

    os.makedirs(args.out, exist_ok=True)
    if args.mvs:
        build_mvs(chunk, os.path.join(args.out, "pointcloud.e57"), crs)
        doc.save()

    # The COLMAP exporter writes images/ and sparse/0/ next to the given file; the file itself is only a readme.
    # First undistorted pinhole (for 3DGS), then the same cameras with their distortion model (no images)
    for folder, pinhole in ((args.out, True), (os.path.join(args.out, "distorted"), False)):
        os.makedirs(folder, exist_ok=True)
        chunk.exportCameras(
            os.path.join(folder, "colmap_export_readme.txt"),
            format=Metashape.CamerasFormatColmap,
            crs=crs,
            convert_to_pinhole=pinhole,
            save_images=pinhole and not args.no_images,
            image_path="{filename}.png",
            binary=False,
            save_markers=False,
        )
    n = export_marker_projections(chunk, os.path.join(args.out, "distorted", "marker_projections.csv"))
    print(f"Exported COLMAP ({args.crs}) to {args.out}, {n} marker projections")
