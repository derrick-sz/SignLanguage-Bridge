import bpy,math
from pathlib import Path
from mathutils import Vector,Matrix
O=Path('/workspace/scratch/87fc5b72c014/godette_v14_exploration')
for tag,fn in [('v14a','checkpoints/godette-v14a-hem-cuff.blend')]:
 bpy.ops.wm.open_mainfile(filepath=str(O/fn),load_ui=False,use_scripts=False);s=bpy.context.scene;r=next(o for o in s.objects if o.type=='ARMATURE')
 for p in r.pose.bones:
  if p.name.startswith(('Body_','Hip_','Spine_','Ribcage_','Neck_','Head_','Clavic.','Arm_','Leg_','Knee.','Foot.','Toe.','Hand.','Finger_','Thumb.')):p.matrix_basis=Matrix.Identity(4)
 bpy.context.view_layer.update()
 for sign,name in [(1,'Arm_Upper_1.L_157'),(-1,'Arm_Upper_1.R_187')]:
  p=r.pose.bones[name];pivot=r.matrix_world@p.head;world=r.matrix_world@p.matrix;rot=Matrix.Translation(pivot)@Matrix.Rotation(sign*math.radians(70),4,'Y')@Matrix.Rotation(-math.pi/2,4,'X')@Matrix.Translation(-pivot);p.matrix=r.matrix_world.inverted()@rot@world;bpy.context.view_layer.update()
 s.cycles.samples=48;s.cycles.use_denoising=False
 views=[('front',(0,-4,0),(0,-.04,.84),1.82,900,1080),('side',(4,0,0),(0,-.04,.84),1.82,900,1080),('back',(0,4,0),(0,-.04,.84),1.82,900,1080),('lower_front',(0,-4,0),(0,-.05,.51),1.07,850,1000)]
 for name,off,target,scale,w,h in views:
  target=Vector(target);c=s.camera;c.location=target+Vector(off);c.rotation_euler=(target-c.location).to_track_quat('-Z','Y').to_euler();c.data.ortho_scale=scale;s.render.resolution_x=w;s.render.resolution_y=h;s.render.filepath=str(O/'previews'/f'{tag}-standing-{name}.png');bpy.ops.render.render(write_still=True);print('READY',tag,name,flush=True)
 if tag=='v14a':bpy.ops.wm.save_as_mainfile(filepath=str(O/'checkpoints/godette-v14a-reference-standing.blend'),compress=True)
