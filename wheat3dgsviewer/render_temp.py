from gaussian_renderer import render

def render_fn(
    camera_state: nerfview.CameraState,
    img_wh: Tuple[int, int],
    gaussians: GaussianModel,
    pipe: dict,
) -> UInt8[np.ndarray, "H W 3"]:
    
    with torch.no_grad():
        W, H = img_wh
        K = camera_state.get_K(img_wh)
        W2C = np.linalg.inv(camera_state.c2w)
        R = W2C[:3, :3].transpose()
        T = W2C[:3, 3]        
        fx = K[0, 0]
        fy = K[1, 1]
        FoVx = 2 * np.arctan(W / (2 * fx))
        FoVy = 2 * np.arctan(H / (2 * fy))

        camera = Camera(
            colmap_id=-1,
            R=R, T=T,
            FoVx=FoVx, FoVy=FoVy,
            image=None,
            gt_alpha_mask=None,
            image_name="render_view",
            uid=0,
            resolution=(W, H),
            data_device="cuda",
            cx=K[0, 2], cy=K[1, 2],
            fl_x=fx, fl_y=fy,
            meta_only=True
        )
        rendered_output = render(
                 viewpoint_camera=camera.cuda(),
                 pc=gaussians,
                 pipe=pipe,
                 bg_color=torch.zeros(3, dtype=torch.float32, device="cuda"),
                 scaling_modifier=1.0
             )
       img = (rendered_output["render"].detach().cpu().numpy() * 255).astype(np.uint8)
       img = img.transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
       return img

def set_camera_frustums(
        server,
        cam_infos,
        test_infos,
        scale_factor=0.1,
        downsample_factor=8,
        frustum_scale=0.05,
        frustum_axes_length=0.005,
        frustum_axes_radius=0.001
        ):
    """Set up camera frames and frustums following COLMAP convention."""
    camera_handles = {}
    frames = []
    image = None
    
    for idx, cam in enumerate(cam_infos):
        # Determine if camera is test or train
        is_test = cam.image_name in test_infos
        split = 'test' if is_test else 'train'
        color = (255, 0, 0) if split == 'test' else (0, 255, 0) # red or green
        
        # Get camera pose
        r = Rotation.from_matrix(cam.R)
        _wxyz = r.as_quat()
        wxyz = np.array([_wxyz[3], _wxyz[0], _wxyz[1], _wxyz[2]], dtype=np.float32)
        position = cam.T * scale_factor
        
        # Add coordinate frame (useful for debugging)
        frame = server.scene.add_frame(
            f"/cameras/{split}/frame_{cam.image_name}",
            wxyz=wxyz,
            position=position,
            axes_length=frustum_axes_length,
            axes_radius=frustum_axes_radius,
            visible=True
        )
        frames.append(frame)

        # Load image if we find it
        if cam.image_path.exists():
            image = iio.imread(cam.image_path)
            image = image[::downsample_factor, ::downsample_factor]     
            
        # Add frustum
        frustum = server.scene.add_camera_frustum(
            f"/cameras/timestep_{timestep}/{split}/frame_{cam.image_name}/frustum", # Note: frustum is now a child of the frame
            fov=2 * np.arctan2(cam.height / 2, cam.fl_y),
            scale=frustum_scale,
            aspect=cam.width / cam.height,
            image=image,
            color=color,
            visible=True,
            #wxyz=wxyz, # wxyz and position set by parent frame
            #position=position,
        )

        # Attach callback to go to the camera when clicked
        def create_camera_callback(frame_position, frame_wxyz, cam_name):
            def callback(event):
                with event.client.atomic():
                    event.client.camera.position = frame_position
                    event.client.camera.wxyz = frame_wxyz
            return callback
        
        frustum.on_click(create_camera_callback(position, wxyz, cam.image_name))
        camera_handles[idx] = frustum
    
    return camera_handles, frames