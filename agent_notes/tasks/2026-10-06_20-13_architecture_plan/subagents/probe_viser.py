"""Print the installed viser version and which scene/gui/server calls exist."""
import viser

print("viser", viser.__version__)
scene_names = [
    "add_mesh_skinned", "add_batched_meshes_simple", "add_batched_meshes_trimesh",
    "add_batched_axes", "add_line_segments", "add_arrows", "add_arrow", "add_camera_frustum",
    "add_point_cloud", "add_transform_controls", "add_gaussian_splats", "add_spline_catmull_rom",
    "add_glb", "add_label", "add_box", "add_cylinder", "add_capsule", "add_icosphere",
    "add_image", "set_environment_map", "configure_default_lights", "add_light_directional",
    "add_light_hemisphere", "configure_fog", "on_pointer_event",
]
names = [n for n in dir(viser.SceneApi) if not n.startswith("_")]
for n in scene_names:
    print(f"scene.{n}: {n in names}")
gui_names = ["add_plotly", "add_uplot", "add_image", "add_markdown", "add_html", "add_progress_bar",
             "configure_theme", "add_tab_group", "add_upload_button", "add_modal"]
gui = [n for n in dir(viser.GuiApi) if not n.startswith("_")]
for n in gui_names:
    print(f"gui.{n}: {n in gui}")
server = [n for n in dir(viser.ViserServer) if not n.startswith("_")]
for n in ["atomic", "get_scene_serializer", "flush", "request_share_url", "on_client_connect"]:
    print(f"server.{n}: {n in server}")
client = [n for n in dir(viser.ClientHandle) if not n.startswith("_")]
print("client.get_render:", "get_render" in client)
print("other scene adds:", sorted(n for n in names if n.startswith("add_") and n not in scene_names))
try:
    import viser.extras as extras
    print("extras:", sorted(n for n in dir(extras) if not n.startswith("_")))
except Exception as error:  # report only
    print("extras import failed:", error)
