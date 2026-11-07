import bpy,json,math
from pathlib import Path
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree
O=Path('/workspace/scratch/87fc5b72c014/godette_v14_exploration');s=bpy.context.scene;r=next(o for o in s.objects if o.type=='ARMATURE');pants=next(o for o in s.objects if 'continuous trousers' in o.name);coat=next(o for o in s.objects if o.name=='Mandarin jacket - front and back');res=[]
def tree(ob):
 e=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());m=e.to_mesh();m.calc_loop_triangles();vs=[e.matrix_world@v.co for v in m.vertices];ts=[tuple(t.vertices) for t in m.loop_triangles];b=BVHTree.FromPolygons(vs,ts,all_triangles=True,epsilon=0);e.to_mesh_clear();return vs,ts,b
def inspect(name,coat_check=False):
 bpy.context.view_layer.update();row={'state':name}
 for obj in ([pants,coat] if coat_check else [pants]):
  vs,ts,b=tree(obj);pairs=[(i,j) for i,j in b.overlap(b) if i<j and not set(ts[i])&set(ts[j])];row['pants' if obj==pants else 'coat']={'nonadjacent_self_candidates':len(pairs),'finite':all(math.isfinite(x) for v in vs for x in v)}
 res.append(row);print(row,flush=True)
def turn(n,d):
 p=r.pose.bones[n];v=r.matrix_world@p.head;w=r.matrix_world@p.matrix;p.matrix=r.matrix_world.inverted()@Matrix.Translation(v)@Matrix.Rotation(math.radians(d),4,'X')@Matrix.Translation(-v)@w;bpy.context.view_layer.update()
def reset(saved):
 for p in r.pose.bones:p.matrix_basis=saved[p.name].copy()
 bpy.context.view_layer.update()
inspect('native_default',True);r.data.pose_position='REST';inspect('rest',True);r.data.pose_position='POSE'
for p in r.pose.bones:
 if p.name.startswith(('Body_','Hip_','Spine_','Ribcage_','Neck_','Head_','Clavic.','Arm_','Leg_','Knee.','Foot.','Toe.','Hand.','Finger_','Thumb.')):p.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update();saved={p.name:p.matrix_basis.copy() for p in r.pose.bones};inspect('neutral_standing',True)
for name,sides,hip,knee in [('left',['L'],35,80),('right',['R'],35,80),('both',['L','R'],25,50)]:
 for i in range(21):
  t=i/20;reset(saved)
  for side in sides:turn('Leg_Upper.L_205' if side=='L' else 'Leg_Upper.R_211',-hip*t);turn('Leg_Lower.L_202' if side=='L' else 'Leg_Lower.R_208',knee*t);turn('Foot.L_201' if side=='L' else 'Foot.R_207',-15*t)
  inspect(name+'_'+str(t),i==20)
report={'candidate':bpy.data.filepath,'method':'Evaluated world-space exact-zero-epsilon triangle BVH; self pairs sharing any vertex excluded. No candidates is sufficient for no triangle self-contact within these sampled poses, not all possible poses.','state_count':len(res),'states':res,'all_pants_zero':all(x['pants']['nonadjacent_self_candidates']==0 for x in res),'all_finite':all(x['pants']['finite'] for x in res)};(O/'validation/v14a-deformation-66-states.json').write_text(json.dumps(report,indent=2));print('DONE',report['state_count'],report['all_pants_zero'],report['all_finite'],flush=True)
