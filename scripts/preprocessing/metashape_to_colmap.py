"""
Camera alignment in Agisoft Metashape and export to COLMAP (undistorted pinhole, centred principal point), in a
projected CRS (default Swiss LV95, EPSG:2056). This is how the reprocessed version of our dataset was made. Afterwards run
recenter_colmap.py, which moves the origin to the mean camera centre (as in the released data) and saves the offset.
Optionally also builds the MVS dense point cloud (--mvs) and exports it as E57 in the same CRS (double precision), and
with --mvs_shift as PLY in that CRS minus a fixed shift (for our data the project coordinates, PCS = LV95 - (2693000,
1256000, 0), as used by the laser scans and scripts/phenotyping/).

Needs Metashape Professional with a license and its Python module
(wheel from https://www.agisoft.com/downloads/installer/, tested with 2.2.1).

Two modes:
    # align the images of a plot, georeference with the markers, save the project and export
    python scripts/preprocessing/metashape_to_colmap.py --images <plot>/images_raw --markers <plot>/plot_461_coordinates.txt \
        --project <out>/agisoft.psx --out <out>/colmap [--mvs]
    # export only, from an existing aligned project (opened read-only, nothing is re-aligned; with --mvs it is
    # opened for writing and the depth maps + dense cloud are saved into it, so use a copy if it isn't yours)
    python scripts/preprocessing/metashape_to_colmap.py --project <out>/agisoft.psx --out <out>/colmap [--no_images]

Markers: one file per plot with label,x,y,z per line in --crs ('#' lines are skipped); labels must match Metashape's
detected coded targets ("target 1", ...). gnss_to_markers.py writes these files from a GNSS (e.g. Trimble) survey.
Coordinates in a local metric system (e.g. tape measurements) work too, with --crs local; see gnss_to_markers.py.
At least 3 targets with coordinates must be found, or the script stops (the export would not be in --crs). Detection
differs between Metashape versions (2.2.1 misses a partly covered target that 2.1 finds): add the pixels of a missed
target with --marker_pixels (CSV with Marker,Camera,X,Y in the raw images, e.g. from another project's
marker_projections.csv or picked by hand).

The export writes <out>/images/<name>.png (undistorted), <out>/sparse/0/{cameras,images,points3D}.txt, the same cameras
with their distortion model (no images; they are the raw input images) in <out>/distorted/sparse/0/, and
<out>/distorted/marker_projections.csv; with --mvs also <out>/pointcloud.e57 (+ pointcloud_shifted.ply). The marker pixels are in the raw
(distorted) input images, as Metashape stores them: use them with the raw images and distorted/sparse/0, not with
images/ and sparse/0. recenter_colmap.py --distorted moves both models by the same offset.

Why pinhole + a projected CRS: the 3DGS camera model here ignores cx, cy (it assumes the principal point at the image
centre), and Metashape's pinhole conversion centres it. Always export with a projected CRS (--crs): without one,
Metashape's COLMAP export uses its own local frame (rotated and shifted relative to the markers' grid), and the
cam_centroid saved by recenter_colmap.py would not map the model back to that CRS.
"""
import csv
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


def add_marker_pixels(chunk, path):
    """Pinned marker positions from a CSV with Marker,Camera,X,Y (raw image pixels, as in marker_projections.csv; '#'
    lines are skipped, more columns are ignored), for targets the detection missed. Rows of cameras not in the chunk
    (e.g. excluded ones) are skipped."""
    cameras = {c.label: c for c in chunk.cameras}
    markers = {m.label: m for m in chunk.markers}
    added = skipped = 0
    with open(path) as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in rows:
            camera = cameras.get(row["Camera"])
            if camera is None:
                skipped += 1
                continue
            if row["Marker"] not in markers:
                markers[row["Marker"]] = chunk.addMarker()
                markers[row["Marker"]].label = row["Marker"]
            markers[row["Marker"]].projections[camera] = Metashape.Marker.Projection(
                Metashape.Vector([float(row["X"]), float(row["Y"])]), True)
            added += 1
    print(f"Added {added} marker pixels from {path} ({skipped} rows of cameras not in the project skipped)")


def marker_labels(markers_path):
    """Labels in a marker coordinates file"""
    with open(markers_path) as f:
        return [line.split(",")[0].strip() for line in f if line.strip() and not line.startswith("#")]


def check_georeferenced(chunk, expected=()):
    """Stop if the chunk isn't georeferenced: then the export would be in Metashape's arbitrary frame, not in --crs.
    expected: labels of the markers file, to name the targets that weren't found at all"""
    used = [m for m in chunk.markers if m.reference.enabled and m.reference.location is not None
            and sum(c.transform is not None for c in m.projections.keys()) >= 2]
    missing = [m.label for m in chunk.markers if m.reference.location is not None and m not in used]
    missing += [label for label in expected if label not in {m.label for m in chunk.markers}]
    if chunk.transform.scale is None or len(used) < 3:
        raise SystemExit(
            f"Not georeferenced: {len(used)} markers with coordinates are seen in >= 2 aligned images "
            f"({[m.label for m in used]}), at least 3 needed. Not found or too few images: {missing}. "
            "If a target is visible but wasn't detected, add its pixels with --marker_pixels and run again.")


def align(images_dir, markers_path, crs, exclude, marker_pixels=None):
    """Same steps as for our reprocessed dataset: detect the coded targets, georeference with their coordinates, align at full resolution"""
    doc = Metashape.Document()
    chunk = doc.addChunk()
    images = sorted(os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".tif")))
    images = [p for p in images if not any(os.path.splitext(p)[0].endswith(f"_{e}") for e in exclude)]
    chunk.addPhotos(images)
    print(f"Added {len(images)} images from {images_dir}")

    chunk.detectMarkers(tolerance=100)
    if marker_pixels is not None:
        add_marker_pixels(chunk, marker_pixels)
    chunk.importReference(markers_path, format=Metashape.ReferenceFormatCSV, delimiter=",", columns="nxyz", crs=crs)
    chunk.crs = crs
    chunk.updateTransform()

    chunk.matchPhotos(downscale=0, reference_preselection_mode=Metashape.ReferencePreselectionSequential)
    chunk.alignCameras()
    return doc, chunk


def build_mvs(chunk, path, crs, shift=None):
    """Dense point cloud as for the paper's MVS clouds (full-resolution depth maps, default filtering), saved as E57.
    With shift, also <out>/pointcloud_shifted.ply with the coordinates in crs minus shift (float32 is precise enough there)"""
    chunk.buildDepthMaps(downscale=1)
    chunk.buildPointCloud()
    chunk.exportPointCloud(path, format=Metashape.PointCloudFormatE57, crs=crs)
    print(f"Exported MVS point cloud to {path}")
    if shift is not None:
        ply = os.path.join(os.path.dirname(path), "pointcloud_shifted.ply")
        # Metashape subtracts the given shift: written coordinates = crs coordinates - shift
        chunk.exportPointCloud(ply, format=Metashape.PointCloudFormatPLY, crs=crs, shift=Metashape.Vector(shift),
                               save_point_color=True, save_point_normal=True, binary=True)
        print(f"Exported MVS point cloud minus {shift} to {ply}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Metashape alignment + COLMAP export in a projected CRS")
    parser.add_argument("--project", type=str, required=True, help="Metashape project (.psx): written with --images, else read")
    parser.add_argument("--out", type=str, required=True, help="output folder for images/, sparse/0/ and marker_projections.csv")
    parser.add_argument("--images", type=str, default=None, help="raw (demosaiced) images to align; omit to export from --project")
    parser.add_argument("--markers", type=str, default=None,
                        help="marker coordinates file (label,x,y,z in --crs), needed with --images; from a GNSS survey: gnss_to_markers.py")
    parser.add_argument("--crs", type=str, default="EPSG::2056",
                        help="CRS of the markers and the export: EPSG code, WKT, or 'local' for local metres (default: Swiss LV95)")
    parser.add_argument("--marker_pixels", type=str, default=None,
                        help="CSV with Marker,Camera,X,Y (raw image pixels) for targets the detection missed, added as pinned (with --images)")
    parser.add_argument("--exclude", type=str, nargs="*", default=["cam_13"], help="image name suffixes to leave out (camera 13 for our plots)")
    parser.add_argument("--no_images", action="store_true", help="only write sparse/ (e.g. when the undistorted images exist already)")
    parser.add_argument("--mvs", action="store_true", help="also build the dense point cloud and export <out>/pointcloud.e57")
    parser.add_argument("--mvs_shift", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="with --mvs, also write <out>/pointcloud_shifted.ply in --crs minus this shift, e.g. 2693000 1256000 0 "
                             "for the project coordinates (PCS) of our TLS clouds and phenotyping scripts")
    parser.add_argument("--gpu_mask", type=int, default=None, help="Metashape GPU bitmask, e.g. 2 = second GPU only (default: Metashape's setting)")
    args = parser.parse_args()

    crs = Metashape.CoordinateSystem(LOCAL_CRS if args.crs == "local" else args.crs)
    if args.gpu_mask is not None:
        Metashape.app.gpu_mask = args.gpu_mask
    if args.images is not None:
        assert args.markers is not None, "--images needs --markers (from a GNSS survey: gnss_to_markers.py)"
        assert not os.path.exists(args.project), f"{args.project} exists; omit --images to export from it"
        doc, chunk = align(args.images, args.markers, crs, args.exclude, args.marker_pixels)
        os.makedirs(os.path.dirname(os.path.abspath(args.project)), exist_ok=True)
        doc.save(args.project)
        print(f"Saved {args.project}")
        # Reopen from disk: depth maps and the dense cloud of a document made in memory are not written into the
        # project by a later doc.save() (seen with 2.2.1), so --mvs would only reach the E57
        del doc, chunk
    else:
        assert args.marker_pixels is None, "--marker_pixels only works with --images (the markers are set before the alignment)"
    doc = Metashape.Document()
    doc.open(args.project, read_only=not args.mvs, ignore_lock=not args.mvs)
    chunk = doc.chunk
    check_georeferenced(chunk, marker_labels(args.markers) if args.markers else ())

    os.makedirs(args.out, exist_ok=True)
    if args.mvs:
        build_mvs(chunk, os.path.join(args.out, "pointcloud.e57"), crs, args.mvs_shift)
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
