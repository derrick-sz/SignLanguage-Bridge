import bpy,math,json,hashlib,sys
from pathlib import Path
from mathutils import Vector
O=Path('/workspace/scratch/87fc5b72c014/godette_v14_exploration');s=bpy.context.scene;records=[]
def sm(a,b,x):
 t=max(0,min(1,(x-a)/(b-a)));return t*t*(3-2*t)
pants=next(o for o in s.objects if 'continuous trousers' in o.name)
# All lower jacket elements share the same world-space warp.
names=('Mandarin jacket','Jade front piping','Ivory inner panel','Jade hem binding','Tailored princess seam','Gold cloud Hem','Gold cloud tail Hem')
for ob in s.objects:
 if ob.type!='MESH':continue
 kind='pants' if ob==pants else 'jacket' if ob.name.startswith(names) else None
 if not kind:continue
 old=[v.co.copy() for v in ob.data.vertices]
 for v in ob.data.vertices:
  p=ob.matrix_world@v.co;x,y,z=p
  if kind=='pants':p.z=z+.018*(1-sm(.088,.29,z))
  else:p.z=z+.012*min(1,(abs(x)/.15)**1.7)*(1-sm(.925,1.10,z))
  v.co=ob.matrix_world.inverted()@p
 ob.data.update();records.append({'mesh':ob.name,'changed_vertices':sum((v.co-a).length>1e-8 for v,a in zip(ob.data.vertices,old)),'max_displacement_m':max((v.co-a).length for v,a in zip(ob.data.vertices,old))})
s['revision_scope']='v14A: reference curved hem and cuff height. Original skeleton, skin mesh, all weights/topology/default poses preserved from frozen v13.'
a=O/'checkpoints/godette-v14a-hem-cuff.blend';bpy.ops.wm.save_as_mainfile(filepath=str(a),compress=True);(O/'validation/v14a-changes.json').write_text(json.dumps({'source':'frozen v13','sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'changes':records},indent=2))
# Separate optional pressed-fabric detail candidate.
old=[v.co.copy() for v in pants.data.vertices]
for v in pants.data.vertices:
 p=pants.matrix_world@v.co;x,y,z=p;t=max(0,min(1,(z-.106)/.764));outer=.118+.029*t*t;inner=.0015+.007*(1-sm(.30,.84,z));cx=(outer+inner)/2
 line=math.exp(-((abs(x)-cx)/.0075)**2);envelope=sm(.105,.17,z)*(1-sm(.73,.87,z));front=sm(.025,.067,-.055-y);back=sm(.025,.066,y+.055)
 p.y=y+line*envelope*(-.00085*front+.00045*back);v.co=pants.matrix_world.inverted()@p
pants.data.update();s['revision_scope']='v14B: v14A plus sub-millimetre pressed trouser creases. Original skeleton, skin mesh, all weights/topology/default poses preserved from frozen v13.'
b=O/'checkpoints/godette-v14b-pressed-crease.blend';bpy.ops.wm.save_as_mainfile(filepath=str(b),compress=True);(O/'validation/v14b-changes.json').write_text(json.dumps({'parent_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'sha256':hashlib.sha256(b.read_bytes()).hexdigest(),'changed_vertices':sum((v.co-a).length>1e-8 for v,a in zip(pants.data.vertices,old)),'max_displacement_m':max((v.co-a).length for v,a in zip(pants.data.vertices,old))},indent=2));print('SAVED',a,b)
