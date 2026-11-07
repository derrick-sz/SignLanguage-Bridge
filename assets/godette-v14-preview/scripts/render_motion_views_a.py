import bpy,math
from pathlib import Path
from mathutils import Matrix,Vector
O=Path('/workspace/scratch/87fc5b72c014/godette_v14_exploration');s=bpy.context.scene;r=next(o for o in s.objects if o.type=='ARMATURE');native={p.name:p.matrix_basis.copy() for p in r.pose.bones}
def render(name,off,target,scale,w=850,h=1000):
 t=Vector(target);c=s.camera;c.location=t+Vector(off);c.rotation_euler=(t-c.location).to_track_quat('-Z','Y').to_euler();c.data.ortho_scale=scale;s.render.resolution_x=w;s.render.resolution_y=h;s.cycles.samples=48;s.cycles.use_denoising=False;s.render.filepath=str(O/'previews'/f'v14a-{name}.png');bpy.ops.render.render(write_still=True);print('READY',name,flush=True)
render('native-default-front',(0,-4,0),(0,-.04,.84),1.82,900,1080)
for p in r.pose.bones:
 if p.name.startswith(('Body_','Hip_','Spine_','Ribcage_','Neck_','Head_','Clavic.','Arm_','Leg_','Knee.','Foot.','Toe.','Hand.','Finger_','Thumb.')):p.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
for sign,name in [(1,'Arm_Upper_1.L_157'),(-1,'Arm_Upper_1.R_187')]:
 p=r.pose.bones[name];v=r.matrix_world@p.head;w=r.matrix_world@p.matrix;p.matrix=r.matrix_world.inverted()@Matrix.Translation(v)@Matrix.Rotation(sign*math.radians(70),4,'Y')@Matrix.Rotation(-math.pi/2,4,'X')@Matrix.Translation(-v)@w;bpy.context.view_layer.update()
saved={p.name:p.matrix_basis.copy() for p in r.pose.bones}
def turn(n,d):
 p=r.pose.bones[n];v=r.matrix_world@p.head;w=r.matrix_world@p.matrix;p.matrix=r.matrix_world.inverted()@Matrix.Translation(v)@Matrix.Rotation(math.radians(d),4,'X')@Matrix.Translation(-v)@w;bpy.context.view_layer.update()
for case,sides,hip,knee in [('standing',[],0,0),('left',['L'],35,80),('right',['R'],35,80),('both',['L','R'],25,50)]:
 for p in r.pose.bones:p.matrix_basis=saved[p.name].copy()
 bpy.context.view_layer.update()
 for side in sides:turn('Leg_Upper.L_205' if side=='L' else 'Leg_Upper.R_211',-hip);turn('Leg_Lower.L_202' if side=='L' else 'Leg_Lower.R_208',knee);turn('Foot.L_201' if side=='L' else 'Foot.R_207',-15)
 for view,off in [('front_oblique',(1.8,-4,0)),('back_oblique',(-1.8,4,0))]:render(case+'-'+view,off,(0,-.04,.60),1.22)
