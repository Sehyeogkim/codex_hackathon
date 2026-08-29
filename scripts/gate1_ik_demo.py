"""Gate 1: prove the arm can execute a pick-and-place under IK control."""
import time

from mimic.rollout import rollout, scripted_pick_place
from mimic.sim import SceneConfig, build_scene
from mimic.video import tile, write_video

cfg = SceneConfig()
model, cfg = build_scene(cfg)
traj = scripted_pick_place(cfg)
print(f"trajectory: {len(traj)} control steps @30Hz = {len(traj)/30:.1f}s")

t0 = time.time()
ep = rollout(model, traj, cfg, cameras=("front", "wrist"), width=480, height=360)
print(f"rollout {time.time()-t0:.1f}s | success={ep.success} lifted={ep.lifted} "
      f"| cube {ep.cube_pos[0].round(3)} -> {ep.cube_pos[-1].round(3)} "
      f"| ik_err max {ep.ik_err.max()*1000:.2f}mm")

path = write_video("out/gate1_pick_place.mp4", tile([ep.frames["front"], ep.frames["wrist"]]), ep.fps)
print("wrote", path)
