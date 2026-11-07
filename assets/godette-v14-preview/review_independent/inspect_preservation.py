import bpy, json, hashlib, math
from pathlib import Path
O=Path('/workspace/scratch/87fc5b72c014/godette_v14_exploration')
R=O/'review_independent'
def h(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',',':')).encode()).hexdigest()
def matrix(m): return [list(x) for x in m]
def plain(v):
 if v is None or isinstance(v,(str,bool,int,float)):return v
 if isinstance(v,set):return sorted(v)
 return [plain(q) for q in v]
def scalar_rna(x):
 out={}
 for p in x.bl_rna.properties:
  k=p.identifier
  if k in {'rna_type','name_full','users','is_updated','is_updated_data','is_updated_transform','is_runtime_data','original','session_uid','bound_box','dimensions'}: continue
  try:
   v=getattr(x,k)
   if p.type in {'BOOLEAN','INT','FLOAT','STRING','ENUM'}:
    if getattr(p,'is_array',False): v=plain(v)
    elif isinstance(v,set):v=sorted(v)
    out[k]=v
   elif p.type=='POINTER': out[k]=None if v is None else getattr(v,'name',type(v).__name__)
  except (AttributeError,TypeError):pass
 return out

def nodes(nt):
 if not nt:return None
 return {'nodes':[{**scalar_rna(n),'inputs':[(i.name,list(i.default_value) if hasattr(i.default_value,'__len__') and not isinstance(i.default_value,str) else i.default_value) for i in n.inputs if hasattr(i,'default_value')]} for n in nt.nodes], 'links':[(l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in nt.links]}

def snap(path):
 bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False)
 s=bpy.context.scene
 out={'meshes':{},'objects':{},'materials':{},'rigs':{},'scene':{'frame':s.frame_current,'frame_start':s.frame_start,'frame_end':s.frame_end,'units':scalar_rna(s.unit_settings),'render':scalar_rna(s.render)},'coords':{}}
 for o in s.objects:
  out['objects'][o.name]={'rna':scalar_rna(o),'matrix_basis':matrix(o.matrix_basis),'matrix_parent_inverse':matrix(o.matrix_parent_inverse),'modifiers':[scalar_rna(m) for m in o.modifiers],'constraints':[scalar_rna(c) for c in o.constraints]}
  if o.type=='MESH':
   m=o.data; gn=[g.name for g in o.vertex_groups]
   out['coords'][o.name]=[(o.matrix_world@v.co).copy() for v in m.vertices]
   out['meshes'][o.name]={'co':h([list(v.co) for v in m.vertices]),'topology':h([list(p.vertices) for p in m.polygons]),'poly_flags':h([(p.material_index,p.use_smooth) for p in m.polygons]),'edges':h([list(e.vertices) for e in m.edges]),'uv':h([(u.name,[list(d.uv) for d in u.data]) for u in m.uv_layers]),'groups':gn,'weights':h([[(gn[g.group],g.weight) for g in v.groups] for v in m.vertices]),'materials':[q.name if q else None for q in m.materials],'shape_keys':h(None if not m.shape_keys else [(k.name,k.value,[list(v.co) for v in k.data]) for k in m.shape_keys.key_blocks])}
  if o.type=='ARMATURE':
   out['rigs'][o.name]={'data_rna':scalar_rna(o.data),'bones':{b.name:{'rna':scalar_rna(b),'matrix':matrix(b.matrix_local),'head':list(b.head_local),'tail':list(b.tail_local)} for b in o.data.bones},'pose':{p.name:{'rna':scalar_rna(p),'matrix_basis':matrix(p.matrix_basis),'constraints':[scalar_rna(c) for c in p.constraints]} for p in o.pose.bones}}
 for m in bpy.data.materials: out['materials'][m.name]={'rna':scalar_rna(m),'nodes':nodes(m.node_tree)}
 return out

def compare(a,b):
 res={'object_names_equal':set(a['objects'])==set(b['objects']),'material_names_equal':set(a['materials'])==set(b['materials']),'scene_equal':a['scene']==b['scene'],'rigs_equal':a['rigs']==b['rigs'],'objects_changed':{},'materials_changed':[],'meshes_changed':{},'pants_regions':{}}
 for n in a['objects']:
  if a['objects'][n]!=b['objects'].get(n):res['objects_changed'][n]={k:([q for q in a['objects'][n][k] if a['objects'][n][k][q]!=b['objects'][n][k].get(q)] if k=='rna' else True) for k in a['objects'][n] if a['objects'][n][k]!=b['objects'][n].get(k)}
 for n in a['materials']:
  if a['materials'][n]!=b['materials'].get(n):res['materials_changed'].append(n)
 for n in a['meshes']:
  if a['meshes'][n]!=b['meshes'].get(n):res['meshes_changed'][n]=[k for k in a['meshes'][n] if a['meshes'][n][k]!=b['meshes'][n].get(k)]
 p=next(n for n in a['coords'] if 'continuous trousers' in n)
 for region,pred in [('all',lambda v:True),('upper_above_0.87',lambda v:v.z>=.87),('central_crotch',lambda v:v.z>.70 and abs(v.x)<.025),('above_cuff_falloff',lambda v:v.z>=.29)]:
  ds=[(v-w).length for v,w in zip(a['coords'][p],b['coords'][p]) if pred(v)]
  res['pants_regions'][region]={'vertices':len(ds),'changed_over_1e8':sum(d>1e-8 for d in ds),'max_displacement':max(ds,default=0)}
 # Check whether any local face degeneracy or flipped local winding normal is introduced.
 bpy.context.view_layer.update()
 obj=bpy.context.scene.objects[p]
 ca=a['coords'][p];cb=b['coords'][p]
 flipped=0;degenerate_new=0;mindot=1.;minarea_ratio=1.
 obj.data.calc_loop_triangles()
 for tri in obj.data.loop_triangles:
  i,j,k=tri.vertices
  na=(ca[j]-ca[i]).cross(ca[k]-ca[i]);nb=(cb[j]-cb[i]).cross(cb[k]-cb[i])
  la,lb=na.length,nb.length
  if la>1e-12:
   if lb<=1e-12:degenerate_new+=1
   else:
    dot=na.normalized().dot(nb.normalized());mindot=min(mindot,dot);flipped+=dot<=0;minarea_ratio=min(minarea_ratio,lb/la)
 res['pants_triangle_check']={'flipped_normal_vs_source':flipped,'newly_degenerate':degenerate_new,'min_corresponding_normal_dot':mindot,'min_area_ratio':minarea_ratio}
 return res
A=snap(O/'baseline/godette-reference-v13.blend')
B=snap(O/'checkpoints/godette-v14a-hem-cuff.blend')
AB=compare(A,B)
C=snap(O/'checkpoints/godette-v14b-pressed-crease.blend')
AC=compare(A,C);BC=compare(B,C)
result={'baseline_units':A['scene']['units'],'baseline_to_A':AB,'baseline_to_B':AC,'A_to_B':BC,'hashes':{str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'baseline/godette-reference-v13.blend',O/'checkpoints/godette-v14a-hem-cuff.blend',O/'checkpoints/godette-v14b-pressed-crease.blend']}}
(R/'preservation-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
