import bpy,json,hashlib
from pathlib import Path
from mathutils import Vector
O=Path('/workspace/scratch/87fc5b72c014/godette_v14_exploration');s=bpy.context.scene;r=next(o for o in s.objects if o.type=='ARMATURE');rows=[]
for ob in s.objects:
 if ob.type!='MESH':continue
 vs=[ob.matrix_world@v.co for v in ob.data.vertices];rows.append({'name':ob.name,'vertices':len(vs),'bounds':[[min(v[i] for v in vs) for i in range(3)],[max(v[i] for v in vs) for i in range(3)]],'materials':[m.name for m in ob.data.materials],'modifiers':[(m.name,m.type) for m in ob.modifiers]})
j={'source':bpy.data.filepath,'sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),'meshes':rows,'bone_count':len(r.data.bones),'camera':s.camera.name,'render_engine':s.render.engine,'revision':s.get('revision_scope')};(O/'validation/baseline-inventory.json').write_text(json.dumps(j,indent=2));print(json.dumps(j,indent=2))
