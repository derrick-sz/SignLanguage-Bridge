import bpy,json
from pathlib import Path
from mathutils.bvhtree import BVHTree
O=Path('/workspace/scratch/87fc5b72c014/godette_v14_exploration');out={};pairs={}
for tag,fn in [('v13','baseline/godette-reference-v13.blend'),('v14a','checkpoints/godette-v14a-hem-cuff.blend'),('v14b','checkpoints/godette-v14b-pressed-crease.blend')]:
 bpy.ops.wm.open_mainfile(filepath=str(O/fn),load_ui=False,use_scripts=False);ob=bpy.data.objects['Mandarin jacket - front and back'];e=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());m=e.to_mesh();m.calc_loop_triangles();vs=[e.matrix_world@v.co for v in m.vertices];ts=[tuple(t.vertices) for t in m.loop_triangles];b=BVHTree.FromPolygons(vs,ts,all_triangles=True,epsilon=0);pairs[tag]=set((i,j) for i,j in b.overlap(b) if i<j and not set(ts[i])&set(ts[j]));pts=[vs[k] for i,j in pairs[tag] for k in ts[i]+ts[j]];out[tag]={'candidate_pairs':len(pairs[tag]),'bbox':[[min(p[a] for p in pts) for a in range(3)],[max(p[a] for p in pts) for a in range(3)]] if pts else None};e.to_mesh_clear()
for tag in ['v14a','v14b']:out[tag]['new_pairs_vs_v13']=len(pairs[tag]-pairs['v13']);out[tag]['removed_pairs_vs_v13']=len(pairs['v13']-pairs[tag]);out[tag]['pair_set_exactly_equal_to_v13']=pairs[tag]==pairs['v13']
(O/'validation/native-coat-contact-comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
