"""Render texture meshes using pytorch3d."""

from turtle import back
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    DirectionalLights,
    BlendParams,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardGouraudShader,
)

background_placeholder = (1, 1, 1)


def image_renderer(
    cam_dist: float = 1200.0,
    cam_elev: float = 0.0,
    cam_azim: float = 0.0,
    camera: str = "FoVPerspectiveCameras",
    fov: float = 15.762,
    imsize: int = 256,
    light_direction: tuple = ((0, 0, 100),),
    ambient_color: tuple = ((0.5, 0.5, 0.5),),
    diffuse_color: tuple = ((0.5, 0.5, 0.5),),
    specular_color: tuple = ((0.05, 0.05, 0.05),),
    background: tuple = (0.5, 0.5, 0.5),
    device: str = None,
    binsize: int = None,
):
    """
    Define the settings for camera, rasterization, and shading

    Args:
        See pytorch3d.renderer.cameras.look_at_view_transform documentation: https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html
            cam_dist
            cam_elev
            cam_azim

        See pytorch3d.renderer.lighting.DirectionalLights documentation: https://pytorch3d.readthedocs.io/en/latest/modules/renderer/lighting.html
            light_direction
            ambient_color
            diffuse_color
            specular_color

        frontal_pose (bool):
            If all faces are generated without head orientations, with certain image size, bin_size is manually set (not based on heuristics) and accelerates rendering.
        background (tuple):
            Image background color. Only one color can be used in this setting.
            See image_rendering() below that generates random background colors.

    Return:
        renderer (pytorch3d.renderer.MeshRenderer)
    """

    if device is None:
        device = torch.device("cuda:0")
    # Transform the shape itself, or transform camera position, or
    R, T = look_at_view_transform(cam_dist, cam_elev, cam_azim)
    # Initialize a perspective camera.
    if camera == "FoVPerspectiveCameras":
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov, degrees=True)
    elif camera == "FoVOrthographicCameras":
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    # Grey background
    if type(background) is not tuple and background.ndim == 3:
        blend_params = BlendParams(background_color=background_placeholder)
    else:
        blend_params = BlendParams(background_color=background)

    # Define the settings for rasterization
    #     print(f'bin_size: {bin_size}')
    raster_settings = RasterizationSettings(
        image_size=imsize,  # Smooth boundary only with 512 * 512
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=binsize,
    )

    lights = DirectionalLights(
        device=device,
        ambient_color=ambient_color,
        diffuse_color=diffuse_color,
        specular_color=specular_color,
        direction=light_direction,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardGouraudShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    return renderer


def image_rendering(
    renderer,
    mesh,
    is_background_random: bool = False,
    background_color: tuple = None,
    device: str = None,
    rgba: bool = False,
):
    """Render image
    Args:
        renderer (pytorch3d.renderer.MeshRenderer)
        mesh (pytorch3d.structures.Meshes)
        is_background_random (bool):
            If True, set different background color (different degrees of gray) for each face.
        background_color (tuple):
            Current background color
        device (str)
    """
    if device is None:
        device = torch.device("cuda:0")

    RGBA = renderer(mesh)  # (N, H, W, 4)
    if is_background_random is True and background_color is None:
        msg = "Currently do not support background color detection. "
        msg += "To set random background, pass in the current background color through the background_color argument."
        raise NotImplementedError(msg)

    if is_background_random is True:
        batch_size, h, w = RGBA.shape[0], RGBA.shape[2], RGBA.shape[3]
        mask = (RGBA[..., :3] == torch.tensor(background_color, device=device)).all(
            3
        )  # N, H, W
        background = (
            torch.rand(batch_size).repeat_interleave(4).view(batch_size, -1).to(device)
        )
        ims = []
        for i_im, (i_mask, i_background) in enumerate(zip(mask, background)):
            im = RGBA[i_im]
            im[i_mask] = i_background
            ims.append(im)
        RGBA = torch.stack(ims)
        del ims

    if is_background_random is False and type(background_color) != tuple:
        assert background_color.ndim == 3, "Pass background image in (H, W, 3)"
        # print('Add background image...')
        batch_size = RGBA.shape[0]
        bckg_mask = (
            (RGBA[..., :3] == torch.tensor(background_placeholder, device=device))
            .all(3)
            .int()
        )  # N, H, W
        im_mask = abs(1 - bckg_mask)  # N, H, W
        background_color = background_color.repeat(batch_size, 1, 1, 1).to(
            device
        )  # N, H, W, 3
        # print(bckg_mask.shape, im_mask.shape, background_color.shape)
        composite = [
            background_color[..., i_chn] * bckg_mask + RGBA[..., i_chn] * im_mask
            for i_chn in range(3)
        ]
        RGBA = torch.stack(composite, axis=3)
        del composite, background_color, im_mask, bckg_mask

    if rgba is False:
        RGBA = RGBA[:, ..., :3]

    RGBA = torch.clamp(RGBA, 0, 1)
    return RGBA
