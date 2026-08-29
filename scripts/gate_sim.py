"""Gate: the simulated Franka can perform the capture-guide task and pass validation."""
import sys, time
sys.path.insert(0, ".")
from mimic.config import DemoConfig
from mimic.rollout import rollout, scripted_pick_place
from mimic.sim import SceneConfig, build_scene
from mimic.video import tile, write_video

demo = DemoConfig.load()
cfg = SceneConfig.from_demo(demo)
model, cfg = build_scene(cfg)
traj = scripted_pick_place(cfg, demo)
print(f"trajectory {len(traj)} steps @30Hz = {len(traj)/30:.1f}s")

t0 = time.time()
ep = rollout(model, traj, cfg, demo, cameras=("front", "wrist"), width=480, height=360)
print(f"rollout {time.time()-t0:.1f}s")
print(ep.validation.report())
print("PASSED" if ep.validation.passed else "FAILED")
write_video("out/gate_sim.mp4", tile([ep.frames["front"], ep.frames["wrist"]]), ep.fps)
print("wrote out/gate_sim.mp4")
